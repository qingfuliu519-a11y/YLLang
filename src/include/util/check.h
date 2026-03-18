#ifndef YLLANG_CUDA_UTIL_H_
#define YLLANG_CUDA_UTIL_H_
#include "util/panic.h"
#include "util/source_location.h"
namespace yllang {

auto inline CudaCheck(cudaError_t error, const yllang::SourceLocation &location = yllang::SourceLocation::Current())
    -> void {
  if (error != ::cudaSuccess) {
    [[unlikely]];
    Panic(location, "CUDA Error: ", ::cudaGetErrorString(error));
  }
}

auto inline CudaCheck(const yllang::SourceLocation &location = yllang::SourceLocation::Current()) -> void {
  CudaCheck(::cudaGetLastError(), location);
}

}  // namespace yllang

#endif  // YLLANG_CUDA_UTIL_H_
