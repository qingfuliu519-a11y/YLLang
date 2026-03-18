/**
 * @file batch_manager.h
 * @brief Defines the BatchManager class responsible for constructing batches from pending requests.
 */

#ifndef YLLANG_BATCH_MANAGER_H
#define YLLANG_BATCH_MANAGER_H

#include <cstddef>
#include <memory>
#include <vector>
#include "batch/batch.h"
#include "decode/decode_manager.h"
#include "kvcache/cache_manager.h"
#include "request/request.h"

namespace yllang {

/**
 * @brief Helper function to construct a flat 1D index tensor from a 2D table and a list of ranges.
 *
 * Each range is a tensor of three elements: [row_start, col_start, col_end).
 * The function collects all linear indices row_start + col (for col in [col_start, col_end))
 * into a single 1D tensor.
 *
 * @param table  2D tensor (unused except for dimension check).
 * @param ranges Vector of 3-element tensors defining the ranges.
 * @return torch::Tensor 1D tensor of linear indices.
 */
static auto Make2DIndices(const torch::Tensor &table, const std::vector<torch::Tensor> &ranges) -> torch::Tensor {
  assert(table.dim() == 2);
  int64_t needle_size = 0;
  for (auto &range : ranges) {
    auto start = range.index({1}).item<size_t>();
    auto end = range.index({2}).item<size_t>();
    needle_size += end - start;
  }
  torch::Tensor indices = torch::empty({needle_size});

  int64_t current_index = 0;
  for (auto &range : ranges) {
    auto row_start = range.index({0}).item<size_t>();
    auto start = row_start + range.index({1}).item<size_t>();
    auto end = row_start + range.index({2}).item<size_t>();
    for (size_t i = start; i < end; i++) {
      indices[current_index++] = i;
    }
  }
  assert(needle_size == current_index);
  return indices;
}

/**
 * @brief Manages the construction of batches from pending requests.
 *
 * Interacts with KVCacheManager to allocate table slots and pages, and performs
 * prefix matching to reuse cached KV data. It respects a prefill budget and
 * produces batches suitable for inference.
 */
class BatchManager {
 public:
  /**
   * @brief Constructs a BatchManager with given cache size and prefill budget.
   *
   * @param max_seq_len     Maximum sequence length (table slots).
   * @param num_pages       Total number of physical pages.
   * @param prefill_budget  Maximum number of new tokens to prefill in a batch.
   */
  BatchManager(const size_t &max_seq_len, const size_t &num_pages, size_t prefill_budget)
      : m_cache_manager_(std::make_shared<KVCacheManager>(num_pages, max_seq_len)),
        m_padding_list_({}),
        m_prefill_budget_(prefill_budget),
        m_reverse_size_(0) {}

  /**
   * @brief Creates the next batch from pending requests.
   *
   * Iterates through the padding list, attempts to add each request to the batch,
   * and constructs the necessary index tensors for loading and writing.
   *
   * @return std::shared_ptr<Batch> The constructed batch, or nullptr if no requests are pending.
   */
  auto NextBatch() -> std::shared_ptr<Batch> {
    if (m_padding_list_.empty()) {
      return nullptr;
    }

    std::vector<std::shared_ptr<Request>> request_list(m_padding_list_.size() << 1);
    for (auto &padding_req : m_padding_list_) {
      auto request = TryAddOneReqToBatch(padding_req);
      if (nullptr == request) {
        break;
      }
      request_list.push_back(request);
    }

    std::vector<torch::Tensor> load_ranges(request_list.size(), torch::empty(3));
    std::vector<torch::Tensor> write_ranges(request_list.size(), torch::empty(1));
    for (size_t i = 0; i < request_list.size(); i++) {
      load_ranges[i][0] = request_list[i]->TableIndex();
      load_ranges[i][1] = request_list[i]->CachedLen();
      load_ranges[i][2] = request_list[i]->DeviceLen();

      write_ranges[i][0] = request_list[i]->DeviceLen();
    }
    const auto load_indices = Make2DIndices(m_cache_manager_->PageTable(), load_ranges);
    const auto write_indices = Make2DIndices(m_cache_manager_->PageTable(), write_ranges);

    auto out_loc = m_cache_manager_->AllocateIndices(load_indices.size(0));
    m_cache_manager_->PageTable().view({-1})[write_indices] = out_loc;

    auto input_ids = m_cache_manager_->TokenPool().view({-1})[load_indices];

    return std::make_shared<Batch>(std::move(request_list), input_ids, out_loc, load_indices, write_indices);
  }

 private:
  /**
   * @brief Attempts to add one pending request to the current batch.
   *
   * Checks for available table slots, performs prefix matching, allocates a table slot,
   * and writes cached tokens if applicable.
   *
   * @param padding_req The pending request.
   * @return std::shared_ptr<Request> A Request object if successfully added, else nullptr.
   */
  auto TryAddOneReqToBatch(const std::shared_ptr<PaddingRequest> &padding_req) -> std::shared_ptr<Request> {
    if (0 < m_cache_manager_->NumTableSlots()) {
      return nullptr;
    }
    auto cache_handle = m_cache_manager_->MatchPrefix(padding_req->InputIds());
    auto cached_len = cache_handle->m_matched_length_;
    auto input_len = padding_req->InputLen();
    auto extend_len = input_len - cached_len;
    if (extend_len + padding_req->OutputLen() > m_cache_manager_->RemainingPageCount()) {
      return nullptr;
    }

    m_cache_manager_->Lock(cache_handle);

    auto table_index = m_cache_manager_->AllocateTableSlot();
    if (cached_len != 0) {
      m_cache_manager_->WriteTokenPool(table_index, padding_req->InputIds().slice(0, 0, cached_len));
      m_cache_manager_->WritePageTable(table_index, cache_handle->m_values_);
    }

    return ChunkPaddingRequest(padding_req, cache_handle, table_index);
  }

  /**
   * @brief Creates a Request object from a padding request, respecting the prefill budget.
   *
   * Splits the input into cached and new parts, updates the prefill budget, and returns
   * a new Request that references the appropriate tokens and cache handle.
   *
   * @param padding_req  The original padding request.
   * @param cache_handle The radix handle from prefix matching.
   * @param table_index  Allocated table slot index.
   * @return std::shared_ptr<Request> The new Request object.
   */
  auto ChunkPaddingRequest(const std::shared_ptr<PaddingRequest> &padding_req,
                           const std::shared_ptr<RadixHandle> &cache_handle, size_t table_index)
      -> std::shared_ptr<Request> {
    auto cached_len = cache_handle->m_matched_length_;
    auto remaining_len = padding_req->InputLen() - cached_len;
    auto extend_len = std::min(remaining_len, m_prefill_budget_);
    m_reverse_size_ += (remaining_len + padding_req->OutputLen());
    m_prefill_budget_ -= extend_len;
    m_cache_manager_->WriteTokenPool(table_index, padding_req->InputIds().slice(0, cached_len, extend_len));
    return std::make_shared<Request>(padding_req->InputIds().slice(0, 0, cached_len + extend_len),
                                     padding_req->UserId(), cached_len, table_index, padding_req->OutputLen(),
                                     cache_handle);
  }

 private:
  std::shared_ptr<KVCacheManager> m_cache_manager_;              ///< Cache manager for page and slot allocation.
  std::shared_ptr<DecodeManager> m_decode_manager_;              ///< Decode manager (currently unused).
  std::vector<std::shared_ptr<PaddingRequest>> m_padding_list_;  ///< List of pending requests.
  size_t m_prefill_budget_;                                      ///< Remaining prefill tokens allowed in current batch.
  size_t m_reverse_size_;                                        ///< Reserved size (usage unclear).
};

}  // namespace yllang

#endif  // YLLANG_BATCH_MANAGER_H