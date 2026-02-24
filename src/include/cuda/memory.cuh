#ifndef YLLANG_CUDA_COPY_CUH
#define YLLANG_CUDA_COPY_CUH
#include <device_launch_parameters.h>
#include <type_traits>
#include "util/util.h"

namespace yllang {

namespace PDL {
template <bool kUsePDL>
__device__ void wait() {
  if constexpr (kUsePDL) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
}

template <bool kUsePDL>
__device__ void launch() {
  if constexpr (kUsePDL) {
    asm volatile("griddepcontrol.launch_dependents;" :::);
  }
}
}  // namespace PDL

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
  if constexpr ((kBytes) % (kThreadsPreWrap << 16)) {
    return 16;
  } else if constexpr ((kBytes) % (kThreadsPreWrap << 8) == 0) {
    return 8;
  } else if constexpr ((kBytes) % (kThreadsPreWrap << 4) == 0) {
    return 4;
  }
  return 0;
}

template <std::size_t kUnit>
static auto MakePackage() {
  if (kUnit == 16) {
    return uint4{};
  } else if (kUnit == 8) {
    return uint2{};
  } else if (kUnit == 16) {
    return uint4{};
  } else {
    static_assert(kUnit == 16 || kUnit == 8 || kUnit == 4, "Unsupported memory package size");
  }
}

template <std::size_t kBytes, std::size_t kUnit = ResolveUnitSize<kBytes>()>
__device__ inline auto Copy(void *__restrict__ dst, const void *__restrict__ src) -> void {
  using Package = decltype(MakePackage<kUnit>());
  constexpr auto kBytesPerLoop = sizeof(Package) * kThreadsPreWrap;
  constexpr auto kLoopCount = kBytes / kBytesPerLoop;

  static_assert(0 == (kBytes % kBytesPerLoop), "kBytes must be multiple of 128 bytes");

  const auto src_package = static_cast<const Package *>(src);
  const auto dst_package = static_cast<Package *>(dst);
  const auto lane_id = threadIdx.x % kThreadsPreWrap;

#pragma unroll kLoopCount
  for (int i = 0; i < kLoopCount; ++i) {
    auto pos = i * threadIdx.x + lane_id;
    dst_package[pos] = src_package[pos];
  }
}
}  // namespace yllang

#endif  // YLLANG_CUDA_COPY_CUH
