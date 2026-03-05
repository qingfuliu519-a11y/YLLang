#ifndef YLLANG_CUDA_UTIL_H_
#define YLLANG_CUDA_UTIL_H_
#include <cuda_runtime.h>
#include <source_location>
#include <sstream>
#include <utility>
#include "util/panic.h"
namespace yllang {

auto inline CudaCheck(cudaError_t error, std::source_location location = std::source_location::current()) -> void {
  if (error != ::cudaSuccess) {
    [[unlikely]];
    Panic(location, "CUDA Error: ", ::cudaGetErrorString(error));
  }
}

auto inline CudaCheck(std::source_location location = std::source_location::current()) -> void {
  CudaCheck(::cudaGetLastError(), location);
}

}  // namespace yllang

#endif  // YLLANG_CUDA_UTIL_H_
