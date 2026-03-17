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

class BatchManager {
 public:
  BatchManager(const size_t &max_seq_len, const size_t &num_pages, size_t prefill_budget)
      : m_cache_manager_(std::make_shared<KVCacheManager>(num_pages, max_seq_len)),
        m_padding_list_({}),
        m_prefill_budget_(prefill_budget),
        m_reverse_size_(0) {}

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
  std::shared_ptr<KVCacheManager> m_cache_manager_;
  std::shared_ptr<DecodeManager> m_decode_manager_;
  std::vector<std::shared_ptr<PaddingRequest>> m_padding_list_;
  size_t m_prefill_budget_;
  size_t m_reverse_size_;
};
}  // namespace yllang

#endif  // YLLANG_BATCH_MANAGER_H
