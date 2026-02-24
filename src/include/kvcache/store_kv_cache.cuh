#ifndef YLLANG_KVCACHE_STORE_KV_CACHE_CUH
#define YLLANG_KVCACHE_STORE_KV_CACHE_CUH

#include <cstdint>
#include <type_traits>
#include "config/config.h"
#include "cuda/memory.cuh"
#include "tvm/ffi/container/tensor.h"
#include "util/util.h"
namespace yllang {

class StoreKernelParams {
 public:
  void *__restrict__ k_cache;
  void *__restrict__ v_cache;
  const void *__restrict__ k;
  const void *__restrict__ v;
  const void *__restrict__ indices;
  const std::size_t kv_cache_stride;
  const std::size_t kv_input_stride;
  const std::size_t length;
};

template <std::size_t kThreadsPreBlock, std::size_t kElementSize, std::integral T, bool kUsePDL>
__device__ auto StoreKVCacheKernel(StoreKernelParams params) -> void {
  constexpr auto kWarpsPreBlock = kThreadsPreBlock / kThreadsPreWrap;
  const auto warp_id = blockIdx.x * kWarpsPreBlock + threadIdx.x / kThreadsPreWrap;

  const auto &[k_cache, v_cache, k, v, indices, kv_cache_stride, kv_input_stride, length] = params;

  yllang::PDL::wait<kUsePDL>();

  if (warp_id < length) {
    const auto pos = static_cast<T *>(indices)[warp_id];
    const auto k_src = yllang::Offset(k, warp_id * kv_input_stride);
    const auto k_dst = yllang::Offset(k_cache, pos * kv_cache_stride);
    yllang::Copy<kElementSize>(k_dst, k_src);

    const auto v_src = yllang::Offset(v, warp_id * kv_input_stride);
    const auto v_dst = yllang::Offset(v_cache, pos * kv_cache_stride);
    yllang::Copy<kElementSize>(v_dst, v_src);
  }

  yllang::PDL::launch<kUsePDL>();
}

template <std::size_t kElementSize, std::size_t kThreadsPreBlock, bool kUsePDL = true>
__global__ auto StoreKVCache(tvm::ffi::TensorView k, tvm::ffi::TensorView v, tvm::ffi::TensorView k_cache,
                             tvm::ffi::TensorView v_cache, tvm::ffi::TensorView indices) -> void {}
}  // namespace yllang

#endif  // YLLANG_KVCACHE_STORE_KV_CACHE_CUH
