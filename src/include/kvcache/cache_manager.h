/**
 * @file cache_manager.h
 * @brief Defines the KVCacheManager class for managing KV cache pages and table slots.
 */

#ifndef YLLANG_KV_CACHE_CACHE_MANAGER_H
#define YLLANG_KV_CACHE_CACHE_MANAGER_H

#include <torch/torch.h>
#include "kvcache/radix_tree.h"

namespace yllang {

/**
 * @brief Manages the KV cache pages, table slots, and token pool.
 *
 * This class is responsible for allocating and recycling table slots and physical pages,
 * maintaining a page table that maps logical positions to physical page indices, and
 * interacting with the radix tree (RadixIndicesManager) for prefix matching and eviction.
 */
class KVCacheManager {
 public:
  /**
   * @brief Constructs a KVCacheManager with given maximum sequence length and number of pages.
   *
   * @param max_sqp_len Maximum number of table slots (logical positions).
   * @param num_pages   Total number of physical pages in the cache.
   */
  KVCacheManager(const size_t &max_sqp_len, const size_t &num_pages)
      : m_max_sqp_len_(max_sqp_len),
        m_num_pages_(num_pages),
        m_indices_manager_(std::make_shared<RadixIndicesManager>()),
        m_table_slots_(torch::range(0, max_sqp_len, 1)),
        m_pages_(torch::range(0, num_pages, 1)),
        m_page_table_(torch::empty({static_cast<int64_t>(num_pages), static_cast<int64_t>(max_sqp_len)})),
        m_token_pool_(torch::empty_like(m_page_table_)) {}

  ~KVCacheManager() = default;

  /**
   * @brief Returns the total number of table slots.
   *
   * @return size_t Number of table slots.
   */
  auto NumTableSlots() const -> size_t { return m_max_sqp_len_; }

  /**
   * @brief Returns the total number of physical pages.
   *
   * @return size_t Number of pages.
   */
  auto NumPages() const -> size_t { return m_num_pages_; }

  /**
   * @brief Returns the remaining number of pages (free pages + evictable pages from radix tree).
   *
   * @return size_t Remaining pages count.
   */
  auto RemainingPageCount() const -> size_t { return m_pages_.size(0) + m_indices_manager_->EvitableSize(); }

  /**
   * @brief Returns the remaining number of table slots.
   *
   * @return size_t Remaining table slots.
   */
  auto RemainingTableSlots() const -> size_t { return m_table_slots_.size(0); }

  /**
   * @brief Returns a const reference to the page table tensor.
   *
   * @return const torch::Tensor& The page table.
   */
  auto PageTable() const -> const torch::Tensor & { return m_page_table_; }

  /**
   * @brief Returns a mutable reference to the page table tensor.
   *
   * @return torch::Tensor The page table.
   */
  auto PageTable() -> torch::Tensor { return m_page_table_; }

  /**
   * @brief Returns a const reference to the token pool tensor.
   *
   * @return const torch::Tensor& The token pool.
   */
  auto TokenPool() const -> const torch::Tensor & { return m_token_pool_; }

  /**
   * @brief Matches a prefix token sequence against the radix tree.
   *
   * @param prefix 1D tensor of token IDs.
   * @return std::shared_ptr<RadixHandle> Handle containing matched information.
   */
  auto MatchPrefix(const torch::Tensor &prefix) const -> std::shared_ptr<RadixHandle> {
    return m_indices_manager_->MatchPrefix(prefix);
  }

  /**
   * @brief Allocates a single table slot.
   *
   * @return size_t The allocated slot index.
   * @throws Torch error if no slots are available.
   */
  auto AllocateTableSlot() -> size_t {
    TORCH_CHECK(RemainingTableSlots() > 0, "no enough table slots");
    const auto slot_index = m_table_slots_.index({-1});
    m_table_slots_ = m_table_slots_.index({torch::indexing::Slice(0, -1)});
    return slot_index.item<size_t>();
  }

  /**
   * @brief Allocates a contiguous chunk of page indices.
   *
   * Tries to allocate from the free page list first; if insufficient, evicts from the radix tree.
   *
   * @param size Number of page indices needed.
   * @return torch::Tensor 1D tensor of allocated page indices.
   */
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

  /**
   * @brief Writes token IDs into the token pool at a given table slot.
   *
   * @param table_index Table slot index.
   * @param token 1D tensor of token IDs to write.
   */
  auto WriteTokenPool(size_t table_index, const torch::Tensor &token) const -> void {
    assert(1 == token.dim());
    auto desition = m_token_pool_.slice(table_index, 0, token.size(0));
    torch::copy(desition, token);
  }

  /**
   * @brief Writes page indices into the page table at a given table slot.
   *
   * @param table_index Table slot index.
   * @param indices 1D tensor of page indices to write.
   */
  auto WritePageTable(size_t table_index, const torch::Tensor &indices) const -> void {
    assert(1 == indices.dim());
    auto desition = m_token_pool_.slice(table_index, 0, indices.size(0));
    torch::copy(desition, indices);
  }

  /**
   * @brief Locks a radix handle, preventing its pages from being evicted.
   *
   * @param handle The handle to lock.
   */
  auto Lock(const std::shared_ptr<RadixHandle> &handle) const -> void {
    assert(nullptr != handle);
    m_indices_manager_->Lock(handle);
  }

  /**
   * @brief Unlocks a radix handle, allowing its pages to become evictable.
   *
   * @param handle The handle to unlock.
   */
  auto Unlock(const std::shared_ptr<RadixHandle> &handle) const -> void {
    assert(nullptr != handle);
    m_indices_manager_->Unlock(handle);
  }

 private:
  size_t m_max_sqp_len_;                                    ///< Maximum number of table slots.
  size_t m_num_pages_;                                      ///< Total number of physical pages.
  std::shared_ptr<RadixIndicesManager> m_indices_manager_;  ///< Radix tree manager for page indices.
  torch::Tensor m_table_slots_;                             ///< 1D tensor of free table slots.
  torch::Tensor m_pages_;                                   ///< 1D tensor of free physical page indices.
  torch::Tensor m_page_table_;                              ///< 2D page table mapping slots to page indices.
  torch::Tensor m_token_pool_;                              ///< 2D token pool storing token IDs per slot.
};

}  // namespace yllang

#endif  // YLLANG_KV_CACHE_CACHE_MANAGER_H