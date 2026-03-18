#ifndef YLLANG_KVCACHE_STORE_KV_CACHE_CUH
#define YLLANG_KVCACHE_STORE_KV_CACHE_CUH

// Standard headers
#include <concepts>
#include <cstdint>

// Project internal headers
#include <torch/torch.h>

namespace yllang {

/**
 * @brief Kernel parameter structure that packs necessary data to be passed to the device kernel.
 *
 * All pointers are qualified with __restrict__ to inform the compiler that they do not alias,
 * which can lead to better optimized code.
 */
class StoreKernelParams {
 public:
  void *__restrict__ m_k_cache_;        // Pointer to the target buffer for K cache
  void *__restrict__ m_v_cache_;        // Pointer to the target buffer for V cache
  const void *__restrict__ m_k_;        // Pointer to the input K tensor data
  const void *__restrict__ m_v_;        // Pointer to the input V tensor data
  const void *__restrict__ m_indices_;  // Pointer to the indices tensor data (type int32_t or int64_t)
  const int64_t m_kv_cache_stride_;     // Stride (in bytes or elements, depending on context) for the cache
  const int64_t m_kv_input_stride_;     // Stride for the input tensor (along the length dimension)
  const int64_t m_length_;              // Number of elements to process (i.e., sequence length)
};

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

#endif  // YLLANG_KVCACHE_STORE_KV_CACHE_CUH