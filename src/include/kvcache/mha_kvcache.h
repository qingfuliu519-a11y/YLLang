/**
 * @file mha_kvcache.h
 * @brief Defines the MHAKVCache class for managing multi-head attention KV cache.
 */

#ifndef YLLANG_KVCACHE_MAH_H_
#define YLLANG_KVCACHE_MAH_H_

#include <torch/torch.h>
#include <vector>
#include "config/config.h"
#include "distributed/info.h"
#include "kvcache/base_kvcache.h"
#include "kvcache/store_kv_cache.cuh"

namespace yllang {

/**
 * @brief Multi-head attention KV cache manager.
 *
 * This class manages the key-value cache for multi-head attention across multiple layers.
 * It handles the storage layout (layer-first or page-first) and provides methods to store
 * KV pairs and access per-layer caches.
 */
class MHAKVCache : public BaseKVCache {
 public:
  /**
   * @brief Constructs an MHAKVCache object.
   *
   * Initializes the KV cache buffer based on the specified layout and dimensions.
   * The buffer is a 6D tensor with dimensions [2, num_layers, num_pages, page_size, num_local_kv_heads, head_dim],
   * where the first dimension separates K and V.
   *
   * @param num_layers   Number of transformer layers.
   * @param num_pages    Number of pages in the cache.
   * @param num_kv_heads Total number of key/value heads (before tensor parallelism).
   * @param head_dim     Dimension of each head.
   * @param kv_layout    Layout of the cache (layer-first or page-first).
   * @param device       Device on which the cache tensors reside.
   * @param dtype        Data type of the cache elements.
   */
  MHAKVCache(int num_layers, int num_pages, int num_kv_heads, int head_dim, KVCacheLayout kv_layout,
             torch::Device device, torch::Dtype dtype)
      : m_num_layers_(num_layers), m_num_pages_(num_pages), m_head_dim_(head_dim), m_device_(device) {
    auto &tp_info = GetDistributedInfo();
    m_num_local_kv_heads_ = num_kv_heads / tp_info.GetSize();
    auto options = torch::TensorOptions().device(device).dtype(dtype);
    switch (kv_layout) {
      case KVCacheLayout::kLayerFirst:
        m_buffer_ = torch::zeros({2, m_num_layers_, m_num_pages_, m_num_local_kv_heads_, m_head_dim_}, options);
        break;
      case KVCacheLayout::kPageFirst:
        m_buffer_ = torch::zeros({2, m_num_layers_, m_num_pages_, m_num_local_kv_heads_, m_head_dim_}, options)
                        .permute({0, 2, 1, 3, 4});
        break;
      default:
        throw std::logic_error("no such KVCacheLayout");
    }
    m_buffer_ = m_buffer_.view({2, m_num_layers_, m_num_pages_, m_page_size_, m_num_local_kv_heads_, m_head_dim_});
    m_k_buffer_ = m_buffer_[0];
    m_v_buffer_ = m_buffer_[1];
    m_view_shape_ = {m_num_pages_ * m_page_size_, -1};
  }

  /**
   * @brief Stores key and value tensors into the cache for a specific layer.
   *
   * This function reshapes the input tensors to 2D (flattening batch/sequence dimensions)
   * and calls the StoreKVCache kernel to copy data into the appropriate cache pages
   * according to the provided location indices.
   *
   * @param k        Input key tensor of shape [batch_size, seq_len, num_heads, head_dim] or similar.
   * @param v        Input value tensor of same shape as k.
   * @param loc      Location indices indicating which pages to store into, shape [batch_size * seq_len].
   * @param layer_id Layer index (0-based).
   */
  auto StoreKV(torch::Tensor k, torch::Tensor v, torch::Tensor loc, int layer_id) -> void override {
    if (loc.size(0) == 0) {
      return;
    }
    auto k_cache = KCache(layer_id);
    auto k_view = k_cache.view(m_view_shape_);

    auto v_cache = VCache(layer_id);
    auto v_view = v_cache.view(m_view_shape_);
    StoreKVCache<kElementSize>(k.view({loc.size(0), -1}), v.view({loc.size(0), -1}), k_view, v_view, loc);
  }

  /**
   * @brief Returns the key cache tensor for a given layer.
   *
   * @param layer_id Layer index (0-based).
   * @return torch::Tensor View of the key cache for the specified layer.
   */
  auto KCache(const int layer_id) -> torch::Tensor override { return m_k_buffer_[layer_id]; }

  /**
   * @brief Returns the value cache tensor for a given layer.
   *
   * @param layer_id Layer index (0-based).
   * @return torch::Tensor View of the value cache for the specified layer.
   */
  auto VCache(const int layer_id) -> torch::Tensor override { return m_v_buffer_[layer_id]; }

  /**
   * @brief Returns the number of layers managed by this cache.
   *
   * @return int Number of layers.
   */
  auto NumLayers() -> int override { return m_num_layers_; }

  /**
   * @brief Returns the device on which the cache tensors are stored.
   *
   * @return torch::Device The device.
   */
  auto Device() -> torch::Device override { return m_device_; }

 private:
  /// Buffer containing both K and V caches, shape [2, num_layers, num_pages, page_size, num_local_kv_heads, head_dim].
  torch::Tensor m_buffer_;

  /// View of the key cache (first slice of m_buffer_).
  torch::Tensor m_k_buffer_;

  /// View of the value cache (second slice of m_buffer_).
  torch::Tensor m_v_buffer_;

  /// Shape used to reshape per-layer cache to [num_pages * page_size, -1] for kernel calls.
  std::vector<int64_t> m_view_shape_;

  /// Number of transformer layers.
  int m_num_layers_;

  /// Number of pages in the cache.
  int m_num_pages_;

  /// Number of local KV heads after tensor parallelism.
  int m_num_local_kv_heads_;

  /// Dimension of each head.
  int m_head_dim_;

  /// Page size (currently fixed to 1).
  int m_page_size_{1};

  /// Device where the cache resides.
  torch::Device m_device_;
};

}  // namespace yllang

#endif  // YLLANG_KVCACHE_MAH_H_