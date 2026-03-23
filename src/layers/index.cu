/**
 * @file index.cu (or index.cuh)
 * @brief Implementation of the embedding lookup kernel.
 */

#include <torch/torch.h>
#include <concepts>
#include <cstdint>
#include "config/config.h"
#include "cuda/launch_kernel.h"
#include "cuda/memory.cuh"
#include "layers/index.h"
#include "util/device.h"
#include "util/tensor.h"

namespace yllang {

/**
 * @brief Kernel parameter structure that packs necessary data to be passed to the device kernel.
 *
 * All pointers are qualified with __restrict__ to inform the compiler that they do not alias,
 * which can lead to better optimized code.
 */
class IndexParams {
 public:
  const void *__restrict__ m_indices_;  ///< Pointer to the indices tensor.
  const void *__restrict__ m_weights_;  ///< Pointer to the embedding weight tensor.
  void *__restrict__ m_destination_;    ///< Pointer to the output destination tensor.
  const int64_t m_weights_stride_;      ///< Stride (in bytes) between rows in the weight tensor.
  const int64_t m_destination_stride_;  ///< Stride (in bytes) between rows in the destination tensor.
  const int64_t m_length_;              ///< Number of indices to process.
};

/**
 * @brief Device-side kernel that performs the embedding lookup.
 *
 * Each warp processes one index. The kernel reads the index, computes the source
 * and destination addresses, and copies the entire row using warp‑cooperative
 * vectorized transfers. PDL (programmatic stream serialization) can be optionally
 * enabled.
 *
 * @tparam kElementSize      Total number of bytes to copy per row.
 * @tparam kThreadsPreBlock  Number of threads per block.
 * @tparam T                 Index type (int32_t or int64_t).
 * @tparam kUsePDL           Whether to use PDL.
 * @param params             Kernel parameter structure.
 */
template <size_t kElementSize, size_t kThreadsPreBlock, std::integral T, bool kUsePDL>
__global__ auto IndexKernel(const __grid_constant__ IndexParams params) -> void {
  // Number of warps per block
  constexpr auto kWarpsPreBlock = kThreadsPreBlock / kThreadsPreWrap;
  // Global warp ID = starting warp of the block + warp ID within the block
  const auto warp_id = (blockIdx.x * kWarpsPreBlock) + (threadIdx.x / kThreadsPreWrap);

  // Structured binding to extract parameter members
  const auto &[indices, weights, destination, weights_stride, destination_stride, length] = params;

  // If PDL is enabled, wait for a condition (used for synchronization or pipelining)
  yllang::pdl::Wait<kUsePDL>();

  // Each warp handles one element; only warps with ID < length execute
  if (std::cmp_less(warp_id, length)) {
    // Read the position from the indices tensor (implicitly cast to type T)
    const auto pos = static_cast<const T *>(indices)[warp_id];

    // Compute source address in the weights table and destination address
    const auto weights_src = yllang::Offset(weights, pos * weights_stride);
    const auto dst = yllang::Offset(destination, warp_id * destination_stride);

    // Copy the entire row using warp‑cooperative vectorized transfers
    yllang::Copy<kElementSize>(dst, weights_src);
  }

  // If PDL is enabled, launch subsequent operations (e.g., release resources)
  yllang::pdl::Launch<kUsePDL>();
}

/**
 * @brief Host-callable kernel launcher that validates tensor shapes and launches IndexKernel.
 *
 * This function uses TensorMatcher to verify the shapes, strides, device, and data types of
 * the input tensors. It then computes the necessary strides (in bytes) and selects the
 * appropriate kernel instantiation based on the index type (int32_t or int64_t). Finally,
 * it launches the kernel using yllang::LaunchKernel.
 *
 * @tparam kElementSize      Total number of bytes to copy per row (must equal hidden_dim * dtype_size).
 * @tparam kThreadsPreBlock  Number of threads per block.
 * @tparam kUsePDL           Whether to use PDL.
 * @param indices           1D tensor of token IDs.
 * @param weights           2D embedding weight tensor.
 * @param destination       2D output tensor.
 */
template <size_t kElementSize, size_t kThreadsPreBlock, bool kUsePDL>
auto Index(const torch::Tensor &indices, const torch::Tensor &weights, const torch::Tensor &destination) -> void {
  // Symbolic variables for shape matching
  SymbolicSize s_element_size{};
  SymbolicSize s_weights_stride{};
  SymbolicSize s_destination_stride{};
  SymbolicSize s_length{};
  SymbolicDType s_type{};
  SymbolicDType s_indices_type{};
  SymbolicDevice s_device{};

  // Validate weights tensor: shape [-1, s_element_size], stride [s_weights_stride, 1]
  TensorMatcher({-1, s_element_size})
      .WithStride({s_weights_stride, 1})
      .WithDevice(s_device)
      .WithDType(s_type)
      .Verify(weights);

  // Validate destination tensor: shape [s_length, s_element_size], stride [s_destination_stride, 1]
  TensorMatcher({s_length, s_element_size})
      .WithStride({s_destination_stride, 1})
      .WithDevice(s_device)
      .WithDType(s_type)
      .Verify(destination);

  // Validate indices tensor: shape [s_length], stride [1]
  TensorMatcher({s_length}).WithStride({1}).WithDevice(s_device).WithDType(s_indices_type).Verify(indices);

  auto num_tokens = s_length.UnWrap();

  constexpr size_t kWrapsPreBlock = kThreadsPreBlock / kThreadsPreWrap;

  // Grid dimension: each block provides kWrapsPreBlock warps, and we need num_tokens warps in total
  dim3 block_dim((num_tokens + kWrapsPreBlock - 1) / kWrapsPreBlock);

  // Construct kernel parameters
  auto params = IndexParams{.m_indices_ = indices.data_ptr(),
                            .m_weights_ = weights.data_ptr(),
                            .m_destination_ = destination.data_ptr(),
                            .m_weights_stride_ = s_weights_stride.UnWrap(),
                            .m_destination_stride_ = s_destination_stride.UnWrap(),
                            .m_length_ = num_tokens};

  bool use_int_32 = (32 == s_indices_type.UnWrap().itemsize());

  // Select the appropriate kernel instantiation based on index type
  const auto kernel = use_int_32 ? yllang::IndexKernel<kElementSize, kThreadsPreBlock, int32_t, kUsePDL>
                                 : yllang::IndexKernel<kElementSize, kThreadsPreBlock, int64_t, kUsePDL>;

  // Launch the kernel
  auto device = yllang::C10Device(s_device.UnWrap());
  yllang::LaunchKernel(block_dim, kThreadsPreBlock, device).WithAttr(kUsePDL)(kernel, params);
}

// Explicit instantiation for the default configuration (kElementSize from config.h)
template auto Index<kElementSize>(const torch::Tensor &indices, const torch::Tensor &weights,
                                  const torch::Tensor &destination) -> void;

}  // namespace yllang