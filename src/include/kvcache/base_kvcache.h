/**
 * @file base_kvcache.h
 * @brief Defines the abstract base class for KV cache implementations.
 */

#ifndef YLLANG_KVCACHE_BASE_H_
#define YLLANG_KVCACHE_BASE_H_

#include <torch/torch.h>

namespace yllang {

/**
 * @brief Layout options for the KV cache tensor.
 */
enum KVCacheLayout : std::uint8_t {
  kPageFirst = 0,  ///< Pages are the first dimension, layers second.
  kLayerFirst      ///< Layers are the first dimension, pages second.
};

/**
 * @brief Abstract base class for key-value cache managers.
 *
 * Derived classes must implement methods to store KV pairs and retrieve per-layer cache tensors.
 */
class BaseKVCache {
 public:
  BaseKVCache() = default;
  virtual ~BaseKVCache() = default;

  /**
   * @brief Stores key and value tensors into the cache for a specific layer.
   *
   * @param k        Key tensor.
   * @param v        Value tensor.
   * @param loc      Location indices (e.g., page indices) where to store.
   * @param layer_id Layer index.
   */
  virtual auto StoreKV(torch::Tensor k, torch::Tensor v, torch::Tensor loc, int layer_id) -> void = 0;

  /**
   * @brief Returns the key cache tensor for a given layer.
   *
   * @param layer_id Layer index.
   * @return torch::Tensor View of the key cache.
   */
  virtual auto KCache(int layer_id) -> torch::Tensor = 0;

  /**
   * @brief Returns the value cache tensor for a given layer.
   *
   * @param layer_id Layer index.
   * @return torch::Tensor View of the value cache.
   */
  virtual auto VCache(int layer_id) -> torch::Tensor = 0;

  /**
   * @brief Returns the number of layers managed.
   *
   * @return int Number of layers.
   */
  virtual auto NumLayers() -> int = 0;

  /**
   * @brief Returns the device where cache tensors reside.
   *
   * @return torch::Device The device.
   */
  virtual auto Device() -> torch::Device = 0;
};

}  // namespace yllang

#endif  // YLLANG_KVCACHE_BASE_H_