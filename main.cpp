#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <random>
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
  std::cout << "\n=== TestStoreKVMultipleRandomWrites ===" << "\n";

  // 配置参数（可根据需要调大以增加压力）
  int num_layers = 100;
  int num_pages = 128;  // 页面总数
  int num_kv_heads = 8;
  int head_dim = 64;
  int max_batch = 3;       // 最大 batch 大小
  int max_seq = 4;         // 最大序列长度
  int num_writes = 10000;  // 写入次数
  torch::Dtype dtype = torch::kFloat32;
  torch::Device device = torch::kCPU;
  if (torch::cuda::is_available()) {
    device = torch::kCUDA;
  }

  yllang::MHAKVCache cache(num_layers, num_pages, num_kv_heads, head_dim, yllang::KVCacheLayout::kPageFirst, device,
                           dtype);
  int layer_id = 0;

  // 记录每个页面最后一次写入的 K 和 V（CPU 副本）
  std::unordered_map<int, std::pair<torch::Tensor, torch::Tensor>> last_written;

  // 随机数生成器
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> batch_dist(1, max_batch);
  std::uniform_int_distribution<int> seq_dist(1, max_seq);

  for (int iter = 0; iter < num_writes; ++iter) {
    // 随机生成 batch 和 seq，但确保总元素数不超过 num_pages
    int batch_size = batch_dist(gen);
    int seq_len = seq_dist(gen);
    while (batch_size * seq_len > num_pages) {
      // 如果超出页面数，重新生成或缩减
      batch_size = batch_dist(gen);
      seq_len = seq_dist(gen);
    }

    // 生成随机输入 K 和 V
    auto k =
        torch::randn({batch_size, seq_len, num_kv_heads, head_dim}, torch::TensorOptions().dtype(dtype).device(device));
    auto v =
        torch::randn({batch_size, seq_len, num_kv_heads, head_dim}, torch::TensorOptions().dtype(dtype).device(device));

    // 生成随机位置索引（确保索引不越界）
    auto loc = RandomLoc(batch_size, seq_len, num_pages, device);

    // 记录本次写入涉及的页面和对应的数据（复制到 CPU 用于后续验证）
    auto loc_cpu = loc.to(torch::kCPU);
    auto k_cpu = k.to(torch::kCPU);
    auto v_cpu = v.to(torch::kCPU);

    // 展平 batch*seq 维度以遍历每个元素
    for (int i = 0; i < batch_size * seq_len; ++i) {
      int page = loc_cpu[i].item<int64_t>();
      // 提取该位置的 K 和 V（形状 [num_kv_heads, head_dim]）
      auto k_i = k_cpu[i / seq_len][i % seq_len].clone();  // 深拷贝到 CPU
      auto v_i = v_cpu[i / seq_len][i % seq_len].clone();
      last_written[page] = {k_i, v_i};  // 覆盖之前记录
    }

    // 执行写入
    cache.StoreKV(k, v, loc, layer_id);
  }
}