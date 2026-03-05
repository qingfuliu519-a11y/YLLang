#ifndef YLLANG_KVCACHE_BASE_H_
#define YLLANG_KVCACHE_BASE_H_

namespace yllang {

enum KVCacheLayout : uint8_t { kPageFirst = 0, kLayerFirst };

class BaseKVCache {
 public:
  BaseKVCache() = default;

  virtual ~BaseKVCache() = default;

  virtual auto StoreKV(torch::Tensor k, torch::Tensor v, torch::Tensor loc, int layer_id) -> void = 0;

  virtual auto KCache(int layer_id) -> torch::Tensor = 0;

  virtual auto VCache(int layer_id) -> torch::Tensor = 0;

  virtual auto NumLayers() -> int = 0;

  virtual auto Device() -> torch::Device = 0;
};
}  // namespace yllang

#endif  // YLLANG_KVCACHE_BASE_H_