#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <random>
#include "models/qwen3.h"
#include "models/model_loader.h"

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
  size_t float32_size = torch::elementSize(torch::kFloat32);
  std::cout << "Size of torch::kFloat32: " << float32_size << " bytes" << '\n';  // 输出 4
}

// 辅助函数：随机生成有效的 loc 张量（确保索引在 [0, num_pages) 内）
auto RandomLoc(int batch_size, int seq_len, int num_pages, torch::Device device) -> torch::Tensor {
  TORCH_CHECK(batch_size * seq_len <= num_pages, "Total sequence length (", batch_size * seq_len,
              ") exceeds number of pages (", num_pages, ")");

  // 生成 0 到 batch_size*seq_len-1 的随机排列（即打乱顺序的连续序列）
  auto loc = torch::randperm(batch_size * seq_len, torch::TensorOptions().dtype(torch::kLong).device(device));
  return loc;
}

auto main(int p, char **v) -> int {
  yllang::ModelLoader::Load(yllang::ModelType::KQwen306b);
  return 0;
}