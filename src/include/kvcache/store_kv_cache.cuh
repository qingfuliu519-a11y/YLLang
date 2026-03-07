#ifndef YLLANG_KVCACHE_STORE_KV_CACHE_CUH
#define YLLANG_KVCACHE_STORE_KV_CACHE_CUH

// Standard headers
#include <concepts>
#include <cstdint>

// Project internal headers
#include <torch/torch.h>
#include "config/config.h"
#include "cuda/launch_kernel.cuh"
#include "cuda/memory.cuh"
#include "util/device.h"
#include "util/tensor.h"

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
 * @brief Device-side kernel that actually performs the KV cache store operation.
 *
 * Each warp is responsible for one element (determined by the index). The function is called
 * on the device and utilizes all threads in a warp to cooperatively copy data (via yllang::Copy).
 *
 * @tparam kElementSize      Total number of bytes to copy (must equal length * kv_dtype_size, checked before call)
 * @tparam kThreadsPreBlock  Number of threads per block
 * @tparam T                 Index type (int32_t or int64_t)
 * @tparam kUsePDL           Whether to use PDL
 * @param params             Kernel parameter structure
 */
template <size_t kElementSize, size_t kThreadsPreBlock, std::integral T, bool kUsePDL>
__global__ auto StoreKVCacheKernel(const __grid_constant__ StoreKernelParams params) -> void {
  // Number of warps per block
  constexpr auto kWarpsPreBlock =
      kThreadsPreBlock / kThreadsPreWrap;  // kThreadsPreWrap is typically 32, defined elsewhere
  // Global warp ID = starting warp of the block + warp ID within the block
  const auto warp_id = (blockIdx.x * kWarpsPreBlock) + (threadIdx.x / kThreadsPreWrap);

  // Structured binding to extract parameter members (C++17 feature)
  const auto &[k_cache, v_cache, k, v, indices, kv_cache_stride, kv_input_stride, length] = params;

  // If PDL is enabled, wait for a condition (possibly used for synchronization or pipelining)
  yllang::pdl::Wait<kUsePDL>();
  // Each warp handles one element; only warps with ID < length execute
  if (std::cmp_less(warp_id, length)) {
    // Read the position from the indices tensor (implicitly cast to type T)
    const auto pos = static_cast<const T *__restrict__>(indices)[warp_id];

    // Compute source and destination addresses for K and perform the copy
    const auto k_src = yllang::Offset(k, warp_id * kv_input_stride);
    const auto k_dst = yllang::Offset(k_cache, pos * kv_cache_stride);
    yllang::Copy<kElementSize>(k_dst, k_src);  // Byte-wise copy, cooperatively done by the whole warp

    // Similarly for V
    const auto v_src = yllang::Offset(v, warp_id * kv_input_stride);
    const auto v_dst = yllang::Offset(v_cache, pos * kv_cache_stride);
    yllang::Copy<kElementSize>(v_dst, v_src);
  }

  // If PDL is enabled, launch subsequent operations (possibly to release resources or trigger events)
  yllang::pdl::Launch<kUsePDL>();
}

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
                  const torch::Tensor &v_cache, const torch::Tensor &indices) -> void {
  // Symbolic variables used to match tensor shapes and attributes (defined in util/tensor.h)
  SymbolicSize s_kv_size{};       // Represents the inner size dimension of each element
  SymbolicSize s_length{};        // Represents the sequence length dimension
  SymbolicSize s_kv_stride{};     // Stride of the input tensor (along the length dimension)
  SymbolicSize s_cache_stride{};  // Stride of the cache tensor (along the length dimension)
  SymbolicDType s_kv_dtype{};     // Data type of K/V
  SymbolicDType indices_type{};   // Data type of the indices tensor
  SymbolicDevice s_device{};      // Device on which the tensors reside

  // Validate cache tensor shape: expected shape [-1, s_kv_size] with stride [s_cache_stride, 1]
  TensorMatcher({-1, s_kv_size})
      .WithStride({s_cache_stride, 1})
      .WithDevice(s_device)
      .WithDType(s_kv_dtype)
      .Verify(k_cache)
      .Verify(v_cache);  // Also verify that v_cache matches k_cache

  // Validate input K/V tensor shape: [s_length, s_kv_size] with stride [s_kv_stride, 1]
  TensorMatcher({s_length, s_kv_size})
      .WithStride({s_kv_stride, 1})
      .WithDevice(s_device)
      .WithDType(s_kv_dtype)
      .Verify(k)
      .Verify(v);

  // Validate indices tensor shape: [s_length] with stride [1]
  TensorMatcher({s_length}).WithStride({1}).WithDevice(s_device).WithDType(indices_type).Verify(indices);

  // Determine if the index type is 32-bit integer
  bool use_int_32 = (32 == indices_type.UnWrap().itemsize());

  // Retrieve the actual length value
  auto element_length = s_kv_size.UnWrap();

  // Get the byte size of the K/V data type
  auto kv_dtype_size = s_kv_dtype.UnWrap().itemsize();

  // Runtime check: the provided kElementSize must equal length * kv_dtype_size
  RuntimeCheck(kElementSize == element_length * kv_dtype_size);

  // Compute number of warps per block
  constexpr auto kWarpsPreBlock = kThreadsPreBlock / kThreadsPreWrap;

  auto num_tokens = s_length.UnWrap();
  // Grid dimension: each block provides kWarpsPreBlock warps, and we need length warps in total
  dim3 block_dim((num_tokens + kWarpsPreBlock - 1) / kWarpsPreBlock);

  // Construct kernel parameters
  auto params = StoreKernelParams{.m_k_cache_ = k_cache.data_ptr(),
                                  .m_v_cache_ = v_cache.data_ptr(),
                                  .m_k_ = k.data_ptr(),
                                  .m_v_ = v.data_ptr(),
                                  .m_indices_ = indices.data_ptr(),
                                  .m_kv_cache_stride_ = static_cast<int64_t>(kv_dtype_size) * s_cache_stride.UnWrap(),
                                  .m_kv_input_stride_ = static_cast<int64_t>(kv_dtype_size) * s_kv_stride.UnWrap(),
                                  .m_length_ = num_tokens};

  // Select the appropriate kernel instantiation based on index type (int32_t or int64_t)
  const auto kernel = use_int_32 ? yllang::StoreKVCacheKernel<kElementSize, kThreadsPreBlock, int32_t, kUsePDL>
                                 : yllang::StoreKVCacheKernel<kElementSize, kThreadsPreBlock, int64_t, kUsePDL>;

  // Launch the kernel using yllang::LaunchKernel (which may handle streams, events, etc.)
  auto device = yllang::C10Device(s_device.UnWrap());

  yllang::LaunchKernel(block_dim, kThreadsPreBlock, device).WithAttr(kUsePDL)(kernel, params);
}

}  // namespace yllang

#endif  // YLLANG_KVCACHE_STORE_KV_CACHE_CUH