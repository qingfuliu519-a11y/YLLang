#include "kvcache/mha_kvcache.h"
#include <torch/torch.h>
#include <cassert>
#include <iostream>
#include <random>

namespace yllang::test {

// 辅助函数：判断两个张量是否在相同设备上
auto SameDevice(const torch::Tensor &a, const torch::Tensor &b) -> bool {
  return a.device().type() == b.device().type() && a.device().index() == b.device().index();
}

// 辅助函数：随机生成有效的 loc 张量（确保索引在 [0, num_pages) 内）
auto RandomLoc(int batch_size, int seq_len, int num_pages, torch::Device device) -> torch::Tensor {
  TORCH_CHECK(batch_size * seq_len <= num_pages, "Total sequence length (", batch_size * seq_len,
              ") exceeds number of pages (", num_pages, ")");

  // 生成 0 到 batch_size*seq_len-1 的随机排列（即打乱顺序的连续序列）
  auto loc = torch::randperm(batch_size * seq_len, torch::TensorOptions().dtype(torch::kLong).device(device));
  return loc;
}

//------------------------------------------------------------------------------
// 测试构造函数在不同布局下是否生成正确的内部缓冲区和外部视图
//------------------------------------------------------------------------------
void TestConstructorCreatesCorrectShapePageFirst() {
  std::cout << "\n=== TestConstructorCreatesCorrectShapePageFirst ===" << "\n";

  int num_layers = 2;
  int num_pages = 4;
  int num_kv_heads = 8;
  int head_dim = 64;
  int page_size = 1;  // 当前固定为1
  torch::Device device = torch::kCPU;
  if (torch::cuda::is_available()) {
    device = torch::kCUDA;
  }
  torch::Dtype dtype = torch::kFloat32;

  MHAKVCache cache(num_layers, num_pages, num_kv_heads, head_dim, KVCacheLayout::kPageFirst, device, dtype);

  for (int layer = 0; layer < num_layers; ++layer) {
    auto k = cache.KCache(layer);
    auto v = cache.VCache(layer);

    std::cout << "Layer " << layer << " K shape: " << k.sizes() << "\n";
    std::cout << "Layer " << layer << " V shape: " << v.sizes() << "\n";

    // 检查维度数应为4
    std::cout << "Checking k.dim() == 4, got " << k.dim() << "\n";
    assert(k.dim() == 4);
    std::cout << "Checking v.dim() == 4, got " << v.dim() << "\n";
    assert(v.dim() == 4);

    // 检查第一维：num_pages
    std::cout << "Checking k.size(0) == " << num_pages << ", got " << k.size(0) << "\n";
    assert(k.size(0) == num_pages);
    std::cout << "Checking v.size(0) == " << num_pages << ", got " << v.size(0) << "\n";
    assert(v.size(0) == num_pages);

    // 检查第二维：page_size（应为1）
    std::cout << "Checking k.size(1) == " << page_size << ", got " << k.size(1) << "\n";
    assert(k.size(1) == page_size);
    std::cout << "Checking v.size(1) == " << page_size << ", got " << v.size(1) << "\n";
    assert(v.size(1) == page_size);

    // 检查第三维：num_kv_heads（假设分布式世界大小为1，本地头数等于全局头数）
    std::cout << "Checking k.size(2) == " << num_kv_heads << ", got " << k.size(2) << "\n";
    assert(k.size(2) == num_kv_heads);
    std::cout << "Checking v.size(2) == " << num_kv_heads << ", got " << v.size(2) << "\n";
    assert(v.size(2) == num_kv_heads);

    // 检查第四维：head_dim
    std::cout << "Checking k.size(3) == " << head_dim << ", got " << k.size(3) << "\n";
    assert(k.size(3) == head_dim);
    std::cout << "Checking v.size(3) == " << head_dim << ", got " << v.size(3) << "\n";
    assert(v.size(3) == head_dim);

    // 验证张量位于正确的设备上
    std::cout << "Checking k.device().type() == " << static_cast<int>(device.type()) << ", got "
              << static_cast<int>(k.device().type()) << "\n";
    assert(k.device().type() == device.type());
    std::cout << "Checking v.device().type() == " << static_cast<int>(device.type()) << ", got "
              << static_cast<int>(v.device().type()) << "\n";
    assert(v.device().type() == device.type());
  }

  std::cout << "--- TestConstructorCreatesCorrectShapePageFirst passed ---\n";
}

void TestConstructorCreatesCorrectShapeLayerFirst() {
  std::cout << "\n=== TestConstructorCreatesCorrectShapeLayerFirst ===" << "\n";

  int num_layers = 2;
  int num_pages = 4;
  int num_kv_heads = 8;
  int head_dim = 64;
  int page_size = 1;
  torch::Device device = torch::kCPU;
  if (torch::cuda::is_available()) {
    device = torch::Device(torch::kCUDA);
  }
  torch::Dtype dtype = torch::kFloat32;

  MHAKVCache cache(num_layers, num_pages, num_kv_heads, head_dim, KVCacheLayout::kLayerFirst, device, dtype);

  for (int layer = 0; layer < num_layers; ++layer) {
    auto k = cache.KCache(layer);
    auto v = cache.VCache(layer);

    std::cout << "Layer " << layer << " K shape: " << k.sizes() << "\n";
    std::cout << "Layer " << layer << " V shape: " << v.sizes() << "\n";

    std::cout << "Checking k.dim() == 4, got " << k.dim() << "\n";
    assert(k.dim() == 4);
    std::cout << "Checking v.dim() == 4, got " << v.dim() << "\n";
    assert(v.dim() == 4);

    std::cout << "Checking k.size(0) == " << num_pages << ", got " << k.size(0) << "\n";
    assert(k.size(0) == num_pages);
    std::cout << "Checking v.size(0) == " << num_pages << ", got " << v.size(0) << "\n";
    assert(v.size(0) == num_pages);

    std::cout << "Checking k.size(1) == " << page_size << ", got " << k.size(1) << "\n";
    assert(k.size(1) == page_size);
    std::cout << "Checking v.size(1) == " << page_size << ", got " << v.size(1) << "\n";
    assert(v.size(1) == page_size);

    std::cout << "Checking k.size(2) == " << num_kv_heads << ", got " << k.size(2) << "\n";
    assert(k.size(2) == num_kv_heads);
    std::cout << "Checking v.size(2) == " << num_kv_heads << ", got " << v.size(2) << "\n";
    assert(v.size(2) == num_kv_heads);

    std::cout << "Checking k.size(3) == " << head_dim << ", got " << k.size(3) << "\n";
    assert(k.size(3) == head_dim);
    std::cout << "Checking v.size(3) == " << head_dim << ", got " << v.size(3) << "\n";
    assert(v.size(3) == head_dim);

    std::cout << "Checking k.device().type() == " << static_cast<int>(device.type()) << ", got "
              << static_cast<int>(k.device().type()) << "\n";
    assert(k.device().type() == device.type());
    std::cout << "Checking v.device().type() == " << static_cast<int>(device.type()) << ", got "
              << static_cast<int>(v.device().type()) << "\n";
    assert(v.device().type() == device.type());
  }

  std::cout << "--- TestConstructorCreatesCorrectShapeLayerFirst passed ---\n";
}

//------------------------------------------------------------------------------
// 测试不同层的 KCache / VCache 是否相互独立（地址不同）
//------------------------------------------------------------------------------
void TestKCacheAndVCacheAreIndependent() {
  std::cout << "\n=== TestKCacheAndVCacheAreIndependent ===" << "\n";

  int num_layers = 2;
  int num_pages = 4;
  int num_kv_heads = 8;
  int head_dim = 64;
  torch::Device device = torch::kCPU;
  if (torch::cuda::is_available()) {
    device = torch::kCUDA;
  }
  torch::Dtype dtype = torch::kFloat32;

  MHAKVCache cache(num_layers, num_pages, num_kv_heads, head_dim, KVCacheLayout::kPageFirst, device, dtype);

  auto k0 = cache.KCache(0);
  auto k1 = cache.KCache(1);
  auto v0 = cache.VCache(0);
  auto v1 = cache.VCache(1);

  std::cout << "k0 data_ptr: " << k0.data_ptr() << ", k1 data_ptr: " << k1.data_ptr() << "\n";
  std::cout << "v0 data_ptr: " << v0.data_ptr() << ", v1 data_ptr: " << v1.data_ptr() << "\n";

  std::cout << "Checking k0.data_ptr() != k1.data_ptr()" << "\n";
  assert(k0.data_ptr() != k1.data_ptr());
  std::cout << "Checking v0.data_ptr() != v1.data_ptr()" << "\n";
  assert(v0.data_ptr() != v1.data_ptr());
  // 确保 K 和 V 也是独立的
  std::cout << "Checking k0.data_ptr() != v0.data_ptr()" << "\n";
  assert(k0.data_ptr() != v0.data_ptr());

  std::cout << "--- TestKCacheAndVCacheAreIndependent passed ---\n";
}

void PrintFlattenedAddresses(const torch::Tensor &k, int batch_size, int seq_len) {
  auto k_flat = k.view({batch_size * seq_len, -1});

  auto sizes = k_flat.sizes();
  auto strides = k_flat.strides();
  size_t elem_size = k_flat.element_size();

  // 打印步长信息表格
  std::cout << "Stride information (original tensor):\n";
  std::cout << std::left << std::setw(10) << "dim" << std::setw(15) << "size" << std::setw(20) << "stride (elements)"
            << "stride (bytes)" << '\n';
  std::cout << std::string(65, '-') << '\n';
  for (size_t i = 0; i < sizes.size(); ++i) {
    std::cout << std::left << std::setw(10) << i << std::setw(15) << sizes[i] << std::setw(20) << strides[i]
              << strides[i] * elem_size << '\n';
  }
  std::cout << '\n';  // 空行分隔

  // 将张量视图为 [batch_size * seq_len, -1]

  // 打印表头
  std::cout << "Address table for flattened view:\n";
  std::cout << std::left << std::setw(8) << "batch" << std::setw(8) << "seq"
            << "address (device)" << '\n';
  std::cout << std::string(40, '-') << '\n';

  // 以十六进制显示地址
  std::cout << std::hex << std::showbase;

  for (int i = 0; i < batch_size * seq_len; ++i) {
    // 从扁平索引 i 恢复 batch 和 seq
    int b = i / seq_len;
    int s = i % seq_len;

    // 获取第 i 行的数据起始地址
    void *addr = k_flat[i].data_ptr();

    std::cout << std::left << std::setw(8) << b << std::setw(8) << s << addr << '\n';
  }

  // 恢复默认输出格式
  std::cout << std::dec << std::noshowbase;
}
// 假设已有张量 k，形状为 [batch_size, seq_len, num_kv_heads, head_dim]
void PrintAddressTable(const torch::Tensor &k, int batch_size, int seq_len) {
  // 打印表头
  std::cout << std::left << std::setw(8) << "batch" << std::setw(8) << "seq"
            << "address" << '\n';
  std::cout << std::string(32, '-') << '\n';  // 分隔线

  // 设置十六进制输出并显示基前缀（0x）
  std::cout << std::hex << std::showbase;

  for (int b = 0; b < batch_size; ++b) {
    for (int s = 0; s < seq_len; ++s) {
      void *addr = k[b][s].data_ptr();  // 获取子张量首地址
      std::cout << std::left << std::setw(8) << b << std::setw(8) << s << addr << '\n';
    }
  }

  // 恢复默认输出格式（可选）
  std::cout << std::dec << std::noshowbase;
}
//------------------------------------------------------------------------------
// 测试 StoreKV 的基本存储功能：写入后能正确读取
//------------------------------------------------------------------------------
void TestStoreKVStoresCorrectly() {
  std::cout << "\n=== TestStoreKVStoresCorrectly ===" << "\n";

  int num_layers = 2;
  int num_pages = 8;
  int num_kv_heads = 8;
  int head_dim = 64;
  torch::Device device = torch::kCPU;
  if (torch::cuda::is_available()) {
    device = torch::kCUDA;
  }
  torch::Dtype dtype = torch::kFloat32;

  MHAKVCache cache(num_layers, num_pages, num_kv_heads, head_dim, KVCacheLayout::kLayerFirst, device, dtype);

  int layer_id = 0;
  int batch_size = 2;
  int seq_len = 3;

  auto k =
      torch::randn({batch_size, seq_len, num_kv_heads, head_dim}, torch::TensorOptions().dtype(dtype).device(device));

  auto v =
      torch::randn({batch_size, seq_len, num_kv_heads, head_dim}, torch::TensorOptions().dtype(dtype).device(device));

  auto loc = RandomLoc(batch_size, seq_len, num_pages, device);

  std::cout << "loc : " << loc << "\n";
  std::cout << "loc sizes: " << loc.sizes() << "\n";
  std::cout << "loc dim: " << loc.dim() << "\n";

  PrintAddressTable(k, batch_size, seq_len);
  PrintFlattenedAddresses(k, batch_size, seq_len);

  cache.StoreKV(k, v, loc, layer_id);

  auto k_cache = cache.KCache(layer_id);  // shape: [num_pages, 1, num_kv_heads, head_dim]
  auto v_cache = cache.VCache(layer_id);

  for (int b = 0; b < batch_size; ++b) {
    for (int s = 0; s < seq_len; ++s) {
      int page = loc[(b * seq_len) + s].item<int64_t>();
      auto k_input = k[b][s];         // [num_kv_heads, head_dim]
      auto v_input = v[b][s];         // [num_kv_heads, head_dim]
      auto k_cached = k_cache[page];  // [1, num_kv_heads, head_dim]
      auto v_cached = v_cache[page];  // [1, num_kv_heads, head_dim]

      // 去掉 page_size 维度（因为 page_size=1）
      k_cached = k_cached.squeeze(0);
      v_cached = v_cached.squeeze(0);

      bool k_ok = torch::allclose(k_input, k_cached);
      bool v_ok = torch::allclose(v_input, v_cached);

      // std::cout << "Batch " << b << ", Seq " << s << ", page " << page << " -> K1 Shape: " << k_input.sizes()
      //           << " K1 dataPtr: " << k_input.data_ptr() << "\n";
      // std::cout << "Batch " << b << ", Seq " << s << ", page " << page << " -> K2 Shape:  : " << k_cached.sizes()
      //           << " K2 dataPtr: " << k_cached.data_ptr() << "\n";
      // std::cout << "Batch " << b << ", Seq " << s << ", page " << page << " -> K1 : \n" << k_input << "\n";
      // std::cout << "Batch " << b << ", Seq " << s << ", page " << page << " -> K2 : \n" << k_cached << "\n";

      std::cout << "Batch " << b << ", Seq " << s << ", page " << page << " -> K allclose: " << (k_ok ? "yes" : "NO")
                << ", V allclose: " << (v_ok ? "yes" : "NO") << "\n";

      std::cout << "Checking K allclose for batch " << b << " seq " << s << "\n";
      assert(k_ok);
      std::cout << "Checking V allclose for batch " << b << " seq " << s << "\n";
      assert(v_ok);
    }
  }

  std::cout << "--- TestStoreKVStoresCorrectly passed ---\n";
}

//------------------------------------------------------------------------------
// 测试多次写入同一 page 时的覆盖行为（后写入应覆盖前写入）
//------------------------------------------------------------------------------
void TestStoreKVOverwritesSamePage() {
  std::cout << "\n=== TestStoreKVOverwritesSamePage ===" << "\n";

  int num_layers = 2;
  int num_pages = 8;
  int num_kv_heads = 8;
  int head_dim = 64;
  torch::Device device = torch::kCPU;
  if (torch::cuda::is_available()) {
    device = torch::kCUDA;
  }
  torch::Dtype dtype = torch::kFloat32;

  MHAKVCache cache(num_layers, num_pages, num_kv_heads, head_dim, KVCacheLayout::kPageFirst, device, dtype);

  int layer_id = 0;
  int target_page = 1;  // 我们将反复写入这个 page

  // 第一次写入
  auto k1 = torch::randn({1, 1, num_kv_heads, head_dim}, device).to(dtype);
  auto v1 = torch::randn({1, 1, num_kv_heads, head_dim}, device).to(dtype);
  auto loc1 = torch::full({1}, target_page, torch::kLong).to(device);
  cache.StoreKV(k1, v1, loc1, layer_id);

  // 第二次写入（不同值）
  auto k2 = torch::randn({1, 1, num_kv_heads, head_dim}, device).to(dtype);
  auto v2 = torch::randn({1, 1, num_kv_heads, head_dim}, device).to(dtype);
  auto loc2 = torch::full({1}, target_page, torch::kLong).to(device);
  cache.StoreKV(k2, v2, loc2, layer_id);

  // 读取并验证为第二次的值
  auto k_cache = cache.KCache(layer_id);  // [num_pages, 1, num_kv_heads, head_dim]
  auto v_cache = cache.VCache(layer_id);
  auto k_cached = k_cache[target_page];  // [1, num_kv_heads, head_dim]
  auto v_cached = v_cache[target_page];  // [1, num_kv_heads, head_dim]

  // 去掉 page_size 维度
  k_cached = k_cached.squeeze(1);
  v_cached = v_cached.squeeze(1);

  // k2[0][0] 形状为 [num_kv_heads, head_dim]
  bool k_match = torch::allclose(k2[0][0], k_cached);
  bool v_match = torch::allclose(v2[0][0], v_cached);
  std::cout << "After second write, K matches second: " << (k_match ? "yes" : "NO") << "\n";
  std::cout << "After second write, V matches second: " << (v_match ? "yes" : "NO") << "\n";

  std::cout << "Checking that K matches second write" << "\n";
  assert(k_match);
  std::cout << "Checking that V matches second write" << "\n";
  assert(v_match);

  std::cout << "--- TestStoreKVOverwritesSamePage passed ---\n";
}

//------------------------------------------------------------------------------
// 测试边界情况：空的输入（batch=0 或 seq=0）不应崩溃
//------------------------------------------------------------------------------
void TestStoreKVWithEmptyInput() {
  std::cout << "\n=== TestStoreKVWithEmptyInput ===" << "\n";

  int num_layers = 2;
  int num_pages = 8;
  int num_kv_heads = 8;
  int head_dim = 64;
  torch::Device device = torch::kCPU;
  if (torch::cuda::is_available()) {
    device = torch::kCUDA;
  }
  torch::Dtype dtype = torch::kFloat32;

  MHAKVCache cache(num_layers, num_pages, num_kv_heads, head_dim, KVCacheLayout::kPageFirst, device, dtype);

  int layer_id = 0;

  // batch = 0
  auto k_empty_batch = torch::randn({0, 3, num_kv_heads, head_dim}, device).to(dtype);
  auto v_empty_batch = torch::randn({0, 3, num_kv_heads, head_dim}, device).to(dtype);
  auto loc_empty_batch = RandomLoc(0, 3, num_pages, device);
  std::cout << "Calling StoreKV with batch=0 (should not crash)" << "\n";
  cache.StoreKV(k_empty_batch, v_empty_batch, loc_empty_batch, layer_id);  // 不应崩溃

  // seq = 0
  auto k_empty_seq = torch::randn({2, 0, num_kv_heads, head_dim}, device).to(dtype);
  auto v_empty_seq = torch::randn({2, 0, num_kv_heads, head_dim}, device).to(dtype);
  auto loc_empty_seq = RandomLoc(2, 0, num_pages, device);
  std::cout << "Calling StoreKV with seq=0 (should not crash)" << "\n";
  cache.StoreKV(k_empty_seq, v_empty_seq, loc_empty_seq, layer_id);  // 不应崩溃

  std::cout << "Empty input handled without crash." << "\n";
  std::cout << "--- TestStoreKVWithEmptyInput passed ---\n";
}

//------------------------------------------------------------------------------
// 测试 NumLayers 和 Device 接口返回正确值
//------------------------------------------------------------------------------
void TestNumLayersAndDevice() {
  std::cout << "\n=== TestNumLayersAndDevice ===" << "\n";

  int num_layers = 3;
  int num_pages = 4;
  int num_kv_heads = 8;
  int head_dim = 64;
  torch::Device device = torch::kCPU;
  if (torch::cuda::is_available()) {
    device = torch::kCUDA;
  }
  torch::Dtype dtype = torch::kFloat32;

  MHAKVCache cache(num_layers, num_pages, num_kv_heads, head_dim, KVCacheLayout::kPageFirst, device, dtype);

  std::cout << "NumLayers(): " << cache.NumLayers() << " (expected " << num_layers << ")" << "\n";
  std::cout << "Checking NumLayers() == " << num_layers << "\n";
  assert(cache.NumLayers() == num_layers);

  std::cout << "Device().type(): " << static_cast<int>(cache.Device().type()) << " (expected "
            << static_cast<int>(device.type()) << ")" << "\n";
  std::cout << "Checking Device().type() == " << static_cast<int>(device.type()) << "\n";
  assert(cache.Device().type() == device.type());

  std::cout << "--- TestNumLayersAndDevice passed ---\n";
}

//------------------------------------------------------------------------------
// 测试多次随机写入不同页面（含覆盖）后，缓存中的内容与最后一次写入一致
//------------------------------------------------------------------------------
void TestStoreKVMultipleRandomWrites() {
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

  MHAKVCache cache(num_layers, num_pages, num_kv_heads, head_dim, KVCacheLayout::kPageFirst, device, dtype);
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

  // 所有写入完成后，验证每个被写过页面的内容是否与最后一次写入一致
  auto k_cache_layer = cache.KCache(layer_id);  // [num_pages, 1, num_kv_heads, head_dim]
  auto v_cache_layer = cache.VCache(layer_id);

  bool all_correct = true;
  for (const auto &[page, kv_pair] : last_written) {
    const auto &expected_k = kv_pair.first;  // CPU tensor [num_kv_heads, head_dim]
    const auto &expected_v = kv_pair.second;

    // 从缓存中取出对应页面的数据（设备上）
    auto cached_k = k_cache_layer[page];  // [1, num_kv_heads, head_dim]
    auto cached_v = v_cache_layer[page];  // [1, num_kv_heads, head_dim]

    // 去掉 page_size 维度（固定为1）
    cached_k = cached_k.squeeze(1);  // [num_kv_heads, head_dim]
    cached_v = cached_v.squeeze(1);  // [num_kv_heads, head_dim]

    // 将缓存数据拷贝到 CPU 进行比较
    auto cached_k_cpu = cached_k.to(torch::kCPU);
    auto cached_v_cpu = cached_v.to(torch::kCPU);

    bool k_ok = torch::allclose(expected_k, cached_k_cpu);
    bool v_ok = torch::allclose(expected_v, cached_v_cpu);

    if (!k_ok || !v_ok) {
      std::cout << "Mismatch on page " << page << " K: " << (k_ok ? "ok" : "FAIL") << " V: " << (v_ok ? "ok" : "FAIL")
                << "\n";
      all_correct = false;
    }
  }

  std::cout << "Checking all written pages match last write..." << "\n";
  assert(all_correct);
  std::cout << "--- TestStoreKVMultipleRandomWrites passed ---\n";
}

}  // namespace yllang::test
//------------------------------------------------------------------------------
// 主函数：运行所有测试
//------------------------------------------------------------------------------
auto main() -> int {
  // 注意：如果 CUDA 可用，所有测试将在 GPU 上运行；否则在 CPU 上运行。
  std::cout << "Running MHAKVCache unit tests...\n";

  yllang::test::TestConstructorCreatesCorrectShapeLayerFirst();
  yllang::test::TestConstructorCreatesCorrectShapePageFirst();
  yllang::test::TestKCacheAndVCacheAreIndependent();
  yllang::test::TestStoreKVStoresCorrectly();
  yllang::test::TestStoreKVOverwritesSamePage();
  yllang::test::TestStoreKVWithEmptyInput();
  yllang::test::TestNumLayersAndDevice();
  yllang::test::TestStoreKVMultipleRandomWrites();
  std::cout << "\nAll tests passed!\n";
  return 0;
}