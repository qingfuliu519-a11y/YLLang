#ifndef YLLANG_CUDA_COPY_CUH
#define YLLANG_CUDA_COPY_CUH
#include <device_launch_parameters.h>
#include <concepts>
#include "util/util.h"

namespace yllang {

namespace pdl {
template <bool kUsePDL>
__device__ void Wait() {
  if constexpr (kUsePDL) {
    // asm volatile("griddepcontrol.wait;" ::: "memory");
  }
}

template <bool kUsePDL>
__device__ void Launch() {
  if constexpr (kUsePDL) {
    // asm volatile("griddepcontrol.launch_dependents;" :::);
  }
}
}  // namespace pdl

template <typename T, std::integral... U>
inline __device__ auto Offset(T *ptr, U... offset) -> void * {
  static_assert(std::is_same_v<void, T>, "Pointer arithmetic is only allowed for void* pointers");
  return static_cast<char *>(ptr) + (... + offset);
}

template <typename T, std::integral... U>
inline __device__ auto Offset(const T *ptr, U... offset) -> const void * {
  static_assert(std::is_same_v<void, T>, "Pointer arithmetic is only allowed for void* pointers");
  return static_cast<const char *>(ptr) + (... + offset);
}

template <std::size_t kBytes>
static constexpr auto ResolveUnitSize() -> std::size_t {
  if constexpr (kBytes % (K_THREADS_PRE_WRAP << 16)) {
    return 16;
  } else if constexpr (kBytes % (K_THREADS_PRE_WRAP << 8) == 0) {
    return 8;
  } else if constexpr (kBytes % (K_THREADS_PRE_WRAP << 4) == 0) {
    return 4;
  }
  return 0;
}

template <std::size_t kUnit>
static auto MakePackage() {
  if constexpr (kUnit == 16) {
    return uint4{};
  } else if constexpr (kUnit == 8) {
    return uint2{};
  } else if constexpr (kUnit == 4) {
    return uint1{};
  }
  static_assert(kUnit == 16 || kUnit == 8 || kUnit == 4, "Unsupported memory package size");
}

template <std::size_t kBytes, std::size_t kUnit = ResolveUnitSize<kBytes>()>
__device__ inline auto Copy(void *__restrict__ dst, const void *__restrict__ src) -> void {
  using Package = decltype(MakePackage<kUnit>());
  constexpr auto k_bytes_per_loop = sizeof(Package) * K_THREADS_PRE_WRAP;
  constexpr auto k_loop_count = kBytes / k_bytes_per_loop;

  static_assert(0 == (kBytes % k_bytes_per_loop), "kBytes must be multiple of 128 bytes");

  const auto src_package = static_cast<const Package *>(src);
  const auto dst_package = static_cast<Package *>(dst);
  const auto lane_id = threadIdx.x % K_THREADS_PRE_WRAP;

#pragma unroll k_loop_count
  for (size_t i = 0; i < k_loop_count; ++i) {
    auto pos = (i * threadIdx.x) + lane_id;
    dst_package[pos] = src_package[pos];
  }
}
}  // namespace yllang

#endif  // YLLANG_CUDA_COPY_CUH
