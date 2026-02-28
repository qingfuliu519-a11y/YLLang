#ifndef YLLANG_CUDA_LAUNCHKERNEL_CUH
#define YLLANG_CUDA_LAUNCHKERNEL_CUH
#include <cuda_runtime.h>
#include "cuda/check.cuh"
namespace yllang {
class LaunchKernel {
 private:
  ::cudaLaunchConfig_t m_config;
  ::cudaLaunchAttribute m_attr_cache;

 public:
  //   LaunchKernel(dim3 grid_dim, dim3 block_dim, DLDevice device, std::size_t dynamic_shared_mem_bytes)
  //       : m_config(MakeCudaLaunchConfig(grid_dim, block_dim, (), dynamic_shared_mem_bytes)), m_attr_cache() {}

  LaunchKernel(const dim3 grid_dim, const dim3 block_dim, cudaStream_t stream, std::size_t dynamic_shared_mem_bytes)
      : m_config(MakeCudaLaunchConfig(grid_dim, block_dim, stream, dynamic_shared_mem_bytes)), m_attr_cache() {}

  LaunchKernel(const LaunchKernel &) = delete;
  LaunchKernel &operator=(const LaunchKernel &) = delete;

  template <typename T, typename... Args>
  auto operator()(T &&kernel, Args &&...args) const -> void {
    CUDA_CHECK(::cudaLaunchKernelEx(&m_config, kernel, std::forward<Args>(args)...));
  }

  auto WithAttr(bool use_pdl) -> LaunchKernel & {
    if (use_pdl) {
      m_attr_cache.id = ::cudaLaunchAttributeProgrammaticStreamSerialization;
      m_attr_cache.val.programmaticStreamSerializationAllowed = 1;
      m_config.attrs = &m_attr_cache;
      m_config.numAttrs = 1;
    } else {
      m_config.numAttrs = 0;
    }
    return *this;
  }

 private:
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
