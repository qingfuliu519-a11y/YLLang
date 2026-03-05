#ifndef YLLANG_CUDA_LAUNCHKERNEL_CUH
#define YLLANG_CUDA_LAUNCHKERNEL_CUH

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "cuda/check.cuh"
#include "util/device.h"
namespace yllang {

/**
 * @brief A helper class to launch CUDA kernels with a convenient and safe interface.
 *
 * It encapsulates a cudaLaunchConfig_t configuration and provides methods to set launch
 * attributes (e.g., programmatic stream serialization) and finally launch the kernel.
 * The class is non-copyable to prevent accidental duplication of launch configurations.
 */
class LaunchKernel {
 private:
  ::cudaLaunchConfig_t m_config_;  ///< The CUDA launch configuration (grid, block, stream, shared memory, attributes)
  ::cudaLaunchAttribute
      m_attr_cache_;  ///< Cached attribute for programmatic stream serialization (used when use_pdl is true)

 public:
  /**
   * @brief Constructs a LaunchKernel from grid/block dimensions and a DLDevice.
   *
   * The CUDA stream is resolved from the given device using TVM's environment API.
   *
   * @param grid_dim                 Grid dimensions in blocks.
   * @param block_dim                Block dimensions in threads.
   * @param device                   DLPack device descriptor.
   * @param dynamic_shared_mem_bytes  Amount of dynamically allocated shared memory per block (default 0).
   */
  LaunchKernel(dim3 grid_dim, dim3 block_dim, yllang::Device &device, std::size_t dynamic_shared_mem_bytes = 0)
      : m_config_(MakeCudaLaunchConfig(grid_dim, block_dim, ResolveDevice(device), dynamic_shared_mem_bytes)),
        m_attr_cache_() {}

  /**
   * @brief Constructs a LaunchKernel from grid/block dimensions and an explicit CUDA stream.
   *
   * @param grid_dim                 Grid dimensions in blocks.
   * @param block_dim                Block dimensions in threads.
   * @param stream                   CUDA stream to use for kernel launch.
   * @param dynamic_shared_mem_bytes  Amount of dynamically allocated shared memory per block (default 0).
   */
  LaunchKernel(const dim3 grid_dim, const dim3 block_dim, cudaStream_t stream, std::size_t dynamic_shared_mem_bytes = 0)
      : m_config_(MakeCudaLaunchConfig(grid_dim, block_dim, stream, dynamic_shared_mem_bytes)), m_attr_cache_() {}

  // Deleted copy constructor and assignment operator to enforce uniqueness of launch configuration.
  LaunchKernel(const LaunchKernel &) = delete;
  auto operator=(const LaunchKernel &) = delete;

  /**
   * @brief Resolves a DLPack device to the corresponding CUDA stream via TVM's FFI.
   *
   * @param device DLPack device descriptor.
   * @return cudaStream_t The CUDA stream associated with the device.
   */
  static auto ResolveDevice(yllang::Device &device) -> cudaStream_t {
    return c10::cuda::getCurrentCUDAStream(device.GetDeviceId());
  }

  /**
   * @brief Function call operator to launch the kernel with the configured parameters.
   *
   * @tparam T    Kernel function type (usually a __global__ function or a lambda).
   * @tparam Args Argument types forwarded to the kernel.
   * @param kernel The kernel entry point.
   * @param args   Arguments passed to the kernel.
   */
  template <typename T, typename... Args>
  auto operator()(T &&kernel, Args &&...args) const -> void {
    CudaCheck(::cudaLaunchKernelEx(&m_config_, kernel, std::forward<Args>(args)...));
  }

  /**
   * @brief Enables or disables programmatic stream serialization attribute.
   *
   * If use_pdl is true, the launch configuration is updated to include the
   * cudaLaunchAttributeProgrammaticStreamSerialization attribute, allowing
   * the kernel to be part of a serialized operation (e.g., PDL). Otherwise,
   * the attribute is cleared.
   *
   * @param use_pdl Flag indicating whether to enable programmatic stream serialization.
   * @return LaunchKernel& Reference to this object (for chaining).
   */
  auto WithAttr(bool use_pdl) -> LaunchKernel & {
    if (use_pdl) {
      m_attr_cache_.id = ::cudaLaunchAttributeProgrammaticStreamSerialization;
      m_attr_cache_.val.programmaticStreamSerializationAllowed = 1;
      m_config_.attrs = &m_attr_cache_;
      m_config_.numAttrs = 1;
    } else {
      m_config_.numAttrs = 0;
    }
    return *this;
  }

 private:
  /**
   * @brief Creates a cudaLaunchConfig_t structure from the given parameters.
   *
   * @param grid_dim                 Grid dimensions.
   * @param block_dim                Block dimensions.
   * @param stream                   CUDA stream.
   * @param dynamic_shared_mem_bytes  Dynamic shared memory size per block.
   * @return ::cudaLaunchConfig_t    Initialized configuration with numAttrs = 0.
   */
  static auto MakeCudaLaunchConfig(dim3 grid_dim, dim3 block_dim, cudaStream_t stream,
                                   std::size_t dynamic_shared_mem_bytes) -> ::cudaLaunchConfig_t {
    ::cudaLaunchConfig_t config{};
    config.blockDim = block_dim;
    config.gridDim = grid_dim;
    config.stream = stream;
    config.dynamicSmemBytes = dynamic_shared_mem_bytes;
    config.numAttrs = 0;
    return config;
  }
};

}  // namespace yllang

#endif  // YLLANG_CUDA_LAUNCHKERNEL_CUH