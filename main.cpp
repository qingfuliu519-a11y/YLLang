#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>

#include "kvcache/mha_kvcache.h"

void TestLibtorchVersion() {
  if (torch::cuda::cudnn_is_available()) {
    std::cout << "cuDNN is available. \n";
  } else {
    std::cout << "cuDNN is not available. \n";
  }
  if (torch::cuda::is_available()) {
    std::cout << "CUDA is available. \n";
  } else {
    std::cout << "CUDA is not available. \n";
  }
  auto count = static_cast<int>(torch::cuda::device_count());
  std::cout << "Device count count count count: " << count << "\n";
}

auto main(int p, char **v) -> int {
  TestLibtorchVersion();
  torch::Tensor a = torch::randint(10, 20, {3, 2});
  std::cout << a << "\n";
  std::cout << "Hello, world!\n";
}
