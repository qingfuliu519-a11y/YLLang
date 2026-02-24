#ifndef YLLANG_CUDA_UTIL_H_
#define YLLANG_CUDA_UTIL_H_
#include <cuda_runtime.h>
#include <source_location>
#include <sstream>
#include <utility>
#include "copy.cuh"
#include "util/panic.h"
namespace yllang {

auto inline CUDA_CHECK(cudaError_t error, std::source_location location = std::source_location::current()) -> void {
  if (error != ::cudaSuccess) {
    [[unlikely]];
    Panic(std::move(location), "CUDA Error: ", ::cudaGetErrorString(error));
  }
}

auto inline CUDA_CHECK(std::source_location location = std::source_location::current()) -> void {
  CUDA_CHECK(::cudaGetLastError(), std::move(location));
}

}  // namespace yllang

#endif  // YLLANG_CUDA_UTIL_H_
