#ifndef YLLANG_KV_CACHE_CACHE_MANAGER_H
#define YLLANG_KV_CACHE_CACHE_MANAGER_H

#include <torch/torch.h>
#include "kvcache/radix_tree.h"

namespace yllang {
class KVCacheManager {
 public:
  KVCacheManager(const size_t &max_sqp_len, const size_t &num_pages)
      : m_max_sqp_len_(max_sqp_len),
        m_num_pages_(num_pages),
        m_indices_manager_(std::make_shared<RadixIndicesManager>()),
        m_table_slots_(torch::range(0, max_sqp_len, 1)),
        m_pages_(torch::range(0, num_pages, 1)),
        m_page_table_(torch::empty({static_cast<int64_t>(num_pages), static_cast<int64_t>(max_sqp_len)})),
        m_token_pool_(torch::empty_like(m_page_table_)) {}

  ~KVCacheManager() = default;

  auto NumTableSlots() const -> size_t { return m_max_sqp_len_; }

  auto NumPages() const -> size_t { return m_num_pages_; }

  auto RemainingPageCount() const -> size_t { return m_pages_.size(0) + m_indices_manager_->EvitableSize(); }

  auto RemainingTableSlots() const -> size_t { return m_table_slots_.size(0); }

  auto PageTable() const -> const torch::Tensor & { return m_page_table_; }

  auto PageTable() -> torch::Tensor { return m_page_table_; }

  auto TokenPool() const -> const torch::Tensor & { return m_token_pool_; }

  auto MatchPrefix(const torch::Tensor &prefix) const -> std::shared_ptr<RadixHandle> {
    return m_indices_manager_->MatchPrefix(prefix);
  }

  auto AllocateTableSlot() -> size_t {
    TORCH_CHECK(RemainingTableSlots() > 0, "no enough table slots");
    const auto slot_index = m_table_slots_.index({-1});
    m_table_slots_ = m_table_slots_.index({torch::indexing::Slice(0, -1)});
    return slot_index.item<size_t>();
  }

  auto AllocateIndices(size_t size) -> torch::Tensor {
    TORCH_CHECK(RemainingPageCount() >= size, "not enough pages");
    if (m_pages_.size(0) >= size) {
      auto slots = m_pages_.slice(0, -size);
      m_pages_ = m_pages_.slice(0, size);
      return slots;
    }
    auto needle_size = size - m_pages_.size(0);
    auto slots = torch::cat({m_pages_, m_indices_manager_->Evict(needle_size)}, 0);
    m_pages_ = m_pages_.slice(0, 0, 0);
    return slots;
  }

  auto WriteTokenPool(size_t table_index, const torch::Tensor &token) const -> void {
    assert(1 == token.dim());
    auto desition = m_token_pool_.slice(table_index, 0, token.size(0));
    torch::copy(desition, token);
  }

  auto WritePageTable(size_t table_index, const torch::Tensor &indices) const -> void {
    assert(1 == indices.dim());
    auto desition = m_token_pool_.slice(table_index, 0, indices.size(0));
    torch::copy(desition, indices);
  }

  auto Lock(const std::shared_ptr<RadixHandle> &handle) const -> void {
    assert(nullptr != handle);
    m_indices_manager_->Lock(handle);
  }

  auto Unlock(const std::shared_ptr<RadixHandle> &handle) const -> void {
    assert(nullptr != handle);
    m_indices_manager_->Unlock(handle);
  }

 private:
  size_t m_max_sqp_len_;
  size_t m_num_pages_;
  std::shared_ptr<RadixIndicesManager> m_indices_manager_;
  torch::Tensor m_table_slots_;
  torch::Tensor m_pages_;
  torch::Tensor m_page_table_;
  torch::Tensor m_token_pool_;
};
}  // namespace yllang

#endif  // YLLANG_KV_CACHE_CACHE_MANAGER_H
