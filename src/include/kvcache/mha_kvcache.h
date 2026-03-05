#ifndef YLLANG_KVCACHE_MAH_H_
#define YLLANG_KVCACHE_MAH_H_

#include <torch/torch.h>
#include <vector>
#include "config/config.h"
#include "distributed/info.h"
#include "kvcache/base_kvcache.h"
#include "kvcache/store_kv_cache.cuh"
namespace yllang {

class MHAKVCache : public BaseKVCache {
 public:
  MHAKVCache(int num_layers, int num_pages, int num_kv_heads, int head_dim, KVCacheLayout kv_layout,
             torch::Device device, torch::Dtype dtype)
      : m_num_layers_(num_layers), m_num_pages_(num_pages), m_head_dim_(head_dim), m_device_(device) {
    auto &tp_info = GetDistributedInfo();
    m_num_local_kv_heads_ = num_kv_heads / tp_info.GetSize();

    auto options = torch::TensorOptions().device(device).dtype(dtype);
    switch (kv_layout) {
      case KVCacheLayout::kLayerFirst:
        m_buffer_ = torch::zeros({2, m_num_layers_, m_num_pages_, m_num_local_kv_heads_, m_head_dim_}, options)
                        .permute({0, 2, 1, 3, 4});
      case KVCacheLayout::kPageFirst:
        m_buffer_ = torch::zeros({2, m_num_pages_, m_num_layers_, m_num_local_kv_heads_, m_head_dim_}, options);
      default:
        throw std::logic_error("no such KVCacheLayout");
    }
    m_k_buffer_ = m_buffer_[0];
    m_v_buffer_ = m_buffer_[1];
    m_view_shape_ = {m_num_pages_, m_num_local_kv_heads_, m_head_dim_};
  }

  auto StoreKV(torch::Tensor k, torch::Tensor v, torch::Tensor loc, int layer_id) -> void override {
    auto k_cache = KCache(layer_id);
    auto k_view = k_cache.view({m_num_pages_, -1});

    auto v_cache = VCache(layer_id);
    auto v_view = v_cache.view({m_num_pages_, -1});
    StoreKVCache<kElementSize>(k, v, k_view, v_view, loc);
  }

  auto KCache(const int layer_id) -> torch::Tensor override { return m_k_buffer_[layer_id]; }

  auto VCache(const int layer_id) -> torch::Tensor override { return m_v_buffer_[layer_id]; }

  auto NumLayers() -> int override { return m_num_layers_; }

  auto Device() -> torch::Device override { return m_device_; }

 private:
  torch::Tensor m_buffer_;
  torch::Tensor m_k_buffer_;
  torch::Tensor m_v_buffer_;
  std::vector<int64_t> m_view_shape_;

  int m_num_layers_;
  int m_num_pages_;
  int m_num_local_kv_heads_;
  int m_head_dim_;
  torch::Device m_device_;
};
}  // namespace yllang

#endif  // YLLANG_KVCACHE_MAH_H_