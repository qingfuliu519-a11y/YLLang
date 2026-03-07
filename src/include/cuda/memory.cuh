/**
 * @file cuda_copy.cuh
 * @brief Provides low-level CUDA utilities for optimized memory copy operations.
 *
 * This file contains helper functions and templates for device-side memory operations,
 * including pointer arithmetic, PDL (grid dependency) control placeholders, and
 * warp-cooperative, vectorized memory copy routines. The copy implementation uses
 * compile-time unit size resolution to maximize throughput.
 */

#ifndef YLLANG_CUDA_COPY_CUH
#define YLLANG_CUDA_COPY_CUH

#include <device_launch_parameters.h>
#include <concepts>
#include "util/util.h"

namespace yllang {

/**
 * @brief Placeholder namespace for PDL (grid dependency control) operations.
 *
 * These functions are intended to be used with CUDA's grid dependency launch
 * features (not yet implemented). They provide conditional compilation based on
 * the kUsePDL flag.
 */
namespace pdl {

/**
 * @brief Conditionally waits for dependent grids to complete.
 *
 * @tparam kUsePDL If true, inserts a wait instruction (currently placeholder).
 */
template <bool kUsePDL>
__device__ void Wait() {
  if constexpr (kUsePDL) {
    // asm volatile("griddepcontrol.wait;" ::: "memory");
  }
}

/**
 * @brief Conditionally launches dependent grids.
 *
 * @tparam kUsePDL If true, inserts a launch instruction (currently placeholder).
 */
template <bool kUsePDL>
__device__ void Launch() {
  if constexpr (kUsePDL) {
    // asm volatile("griddepcontrol.launch_dependents;" :::);
  }
}
}  // namespace pdl

/**
 * @brief Advances a void* pointer by a sum of integral offsets.
 *
 * This function is restricted to void* pointers because pointer arithmetic
 * on void* is not standard; the implementation casts to char* for byte-wise
 * offset calculation.
 *
 * @tparam T          Must be void (enforced by static_assert).
 * @tparam U          Integral types for offsets.
 * @param ptr         Base pointer (must be void*).
 * @param offset      One or more integral offsets to sum.
 * @return void*      New pointer offset by total bytes.
 */
template <typename T, std::integral... U>
inline __device__ auto Offset(T *ptr, U... offset) -> void * {
  static_assert(std::is_same_v<void, T>, "Pointer arithmetic is only allowed for void* pointers");
  return static_cast<char *>(ptr) + (... + offset);
}

/**
 * @brief Const-qualified version of Offset for const void* pointers.
 *
 * @tparam T          Must be const void (enforced by static_assert).
 * @tparam U          Integral types for offsets.
 * @param ptr         Base const pointer.
 * @param offset      One or more integral offsets to sum.
 * @return const void* New const pointer offset by total bytes.
 */
template <typename T, std::integral... U>
inline __device__ auto Offset(const T *ptr, U... offset) -> const void * {
  static_assert(std::is_same_v<void, T>, "Pointer arithmetic is only allowed for void* pointers");
  return static_cast<const char *>(ptr) + (... + offset);
}

/**
 * @brief Compile-time determination of optimal memory access unit size.
 *
 * Evaluates divisibility of kBytes by powers of two times the warp size
 * (K_THREADS_PRE_WRAP) to select 16-, 8-, or 4-byte units. Returns 0 if no
 * suitable unit is found.
 *
 * @tparam kBytes     Total number of bytes to copy.
 * @return std::size_t Optimal unit size (16, 8, 4, or 0).
 */
template <std::size_t kBytes>
static constexpr auto ResolveUnitSize() -> std::size_t {
  if constexpr (kBytes % (kThreadsPreWrap << 16)) {
    return 16;
  } else if constexpr (kBytes % (kThreadsPreWrap << 8) == 0) {
    return 8;
  } else if constexpr (kBytes % (kThreadsPreWrap << 4) == 0) {
    return 4;
  }
  return 0;
}

/**
 * @brief Returns a CUDA vector type corresponding to the requested unit size.
 *
 * This function is intended to be used only for its return type via decltype.
 * It returns a value of type uint4 (16 bytes), uint2 (8 bytes), or uint1 (4 bytes)
 * depending on kUnit.
 *
 * @tparam kUnit      Unit size (must be 16, 8, or 4).
 * @return An instance of the corresponding vector type.
 */
template <std::size_t kUnit>
__device__ static auto MakePackage() {
  if constexpr (kUnit == 16) {
    return uint4{};
  } else if constexpr (kUnit == 8) {
    return uint2{};
  } else if constexpr (kUnit == 4) {
    return uint1{};
  }
  static_assert(kUnit == 16 || kUnit == 8 || kUnit == 4, "Unsupported memory package size");
}

/**
 * @brief Warp-cooperative, vectorized memory copy from src to dst.
 *
 * The copy is performed using the optimal vector type determined by
 * ResolveUnitSize<kBytes>(). Each thread within a warp copies a portion
 * of the data, and the operation is unrolled for efficiency.
 *
 * @tparam kBytes     Total number of bytes to copy.
 * @tparam kUnit      Unit size (defaults to ResolveUnitSize<kBytes>). Must divide
 *                    (sizeof(Package) * K_THREADS_PRE_WRAP) evenly into kBytes.
 * @param dst         Destination pointer (must point to at least kBytes bytes).
 * @param src         Source pointer (must point to at least kBytes bytes).
 */
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
  for (size_t i = 0; i < kLoopCount; ++i) {
    auto pos = (i * kThreadsPreWrap) + lane_id;
    dst_package[pos] = src_package[pos];
  }
}
}  // namespace yllang

#endif  // YLLANG_CUDA_COPY_CUH