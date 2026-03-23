#ifndef YLLANG_KVCACHE_STORE_KV_CACHE_H
#define YLLANG_KVCACHE_STORE_KV_CACHE_H

// Standard headers
#include <concepts>
#include <cstdint>

// Project internal headers
#include <torch/torch.h>

namespace yllang {

/**
 * @brief Host-callable kernel launcher that validates tensor shapes, computes parameters,
 *        and launches the StoreKVCacheKernel.
 *
 * This function uses TensorMatcher to verify their shapes, strides, device, and data types.
 * It then selects the kernel instantiation based on the index type (int32_t or int64_t) and launches it.
 *
 * @tparam kElementSize      Total number of bytes to copy (must equal length * kv_dtype_size)
 * @tparam kThreadsPreBlock  Number of threads per block
 * @tparam kUsePDL           Whether to use PDL (default true)
 * @param k                  Input K tensor
 * @param v                  Input V tensor
 * @param k_cache            K cache tensor
 * @param v_cache            V cache tensor
 * @param indices            Indices tensor
 */
template <size_t kElementSize, size_t kThreadsPreBlock = 128, bool kUsePDL = false>
auto StoreKVCache(const torch::Tensor &k, const torch::Tensor &v, const torch::Tensor &k_cache,
                  const torch::Tensor &v_cache, const torch::Tensor &indices) -> void;

}  // namespace yllang

#endif  // YLLANG_KVCACHE_STORE_KV_CACHE_H