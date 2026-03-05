#include "kvcache/mha_kvcache.h"
#include <torch/torch.h>
#include <cassert>
#include <iostream>
#include <vector>

namespace yllang {
namespace test {

// 辅助函数：判断两个张量是否在相同设备上
bool SameDevice(const torch::Tensor &a, const torch::Tensor &b) {
  return a.device().type() == b.device().type() && a.device().index() == b.device().index();
}

// 辅助函数：随机生成有效的 loc 张量（确保索引在 [0, num_pages) 内）
torch::Tensor RandomLoc(int batch_size, int seq_len, int num_pages, torch::Device device) {
  return torch::randint(0, num_pages, {batch_size * seq_len},
                        torch::TensorOptions().dtype(torch::kLong).device(device));
}

//------------------------------------------------------------------------------
// 测试构造函数在不同布局下是否生成正确的内部缓冲区和外部视图
//------------------------------------------------------------------------------
void TestConstructorCreatesCorrectShapePageFirst() {
  std::cout << "\n=== TestConstructorCreatesCorrectShapePageFirst ===" << std::endl;

  int num_layers = 2;
  int num_pages = 4;
  int num_kv_heads = 8;
  int head_dim = 64;
  int page_size = 1;  // 当前固定为1
  torch::Device device = torch::kCPU;
  if (torch::cuda::is_available()) device = torch::kCUDA;
  torch::Dtype dtype = torch::kFloat32;

  MHAKVCache cache(num_layers, num_pages, num_kv_heads, head_dim, KVCacheLayout::kPageFirst, device, dtype);

  for (int layer = 0; layer < num_layers; ++layer) {
    auto k = cache.KCache(layer);
    auto v = cache.VCache(layer);

    std::cout << "Layer " << layer << " K shape: " << k.sizes() << std::endl;
    std::cout << "Layer " << layer << " V shape: " << v.sizes() << std::endl;

    // 检查维度数应为4
    std::cout << "Checking k.dim() == 4, got " << k.dim() << std::endl;
    assert(k.dim() == 4);
    std::cout << "Checking v.dim() == 4, got " << v.dim() << std::endl;
    assert(v.dim() == 4);

    // 检查第一维：num_pages
    std::cout << "Checking k.size(0) == " << num_pages << ", got " << k.size(0) << std::endl;
    assert(k.size(0) == num_pages);
    std::cout << "Checking v.size(0) == " << num_pages << ", got " << v.size(0) << std::endl;
    assert(v.size(0) == num_pages);

    // 检查第二维：page_size（应为1）
    std::cout << "Checking k.size(1) == " << page_size << ", got " << k.size(1) << std::endl;
    assert(k.size(1) == page_size);
    std::cout << "Checking v.size(1) == " << page_size << ", got " << v.size(1) << std::endl;
    assert(v.size(1) == page_size);

    // 检查第三维：num_kv_heads（假设分布式世界大小为1，本地头数等于全局头数）
    std::cout << "Checking k.size(2) == " << num_kv_heads << ", got " << k.size(2) << std::endl;
    assert(k.size(2) == num_kv_heads);
    std::cout << "Checking v.size(2) == " << num_kv_heads << ", got " << v.size(2) << std::endl;
    assert(v.size(2) == num_kv_heads);

    // 检查第四维：head_dim
    std::cout << "Checking k.size(3) == " << head_dim << ", got " << k.size(3) << std::endl;
    assert(k.size(3) == head_dim);
    std::cout << "Checking v.size(3) == " << head_dim << ", got " << v.size(3) << std::endl;
    assert(v.size(3) == head_dim);

    // 验证张量位于正确的设备上
    std::cout << "Checking k.device().type() == " << static_cast<int>(device.type()) << ", got "
              << static_cast<int>(k.device().type()) << std::endl;
    assert(k.device().type() == device.type());
    std::cout << "Checking v.device().type() == " << static_cast<int>(device.type()) << ", got "
              << static_cast<int>(v.device().type()) << std::endl;
    assert(v.device().type() == device.type());
  }

  std::cout << "--- TestConstructorCreatesCorrectShapePageFirst passed ---\n";
}

void TestConstructorCreatesCorrectShapeLayerFirst() {
  std::cout << "\n=== TestConstructorCreatesCorrectShapeLayerFirst ===" << std::endl;

  int num_layers = 2;
  int num_pages = 4;
  int num_kv_heads = 8;
  int head_dim = 64;
  int page_size = 1;
  torch::Device device = torch::kCPU;
  if (torch::cuda::is_available()) device = torch::kCUDA;
  torch::Dtype dtype = torch::kFloat32;

  MHAKVCache cache(num_layers, num_pages, num_kv_heads, head_dim, KVCacheLayout::kLayerFirst, device, dtype);

  for (int layer = 0; layer < num_layers; ++layer) {
    auto k = cache.KCache(layer);
    auto v = cache.VCache(layer);

    std::cout << "Layer " << layer << " K shape: " << k.sizes() << std::endl;
    std::cout << "Layer " << layer << " V shape: " << v.sizes() << std::endl;

    std::cout << "Checking k.dim() == 4, got " << k.dim() << std::endl;
    assert(k.dim() == 4);
    std::cout << "Checking v.dim() == 4, got " << v.dim() << std::endl;
    assert(v.dim() == 4);

    std::cout << "Checking k.size(0) == " << num_pages << ", got " << k.size(0) << std::endl;
    assert(k.size(0) == num_pages);
    std::cout << "Checking v.size(0) == " << num_pages << ", got " << v.size(0) << std::endl;
    assert(v.size(0) == num_pages);

    std::cout << "Checking k.size(1) == " << page_size << ", got " << k.size(1) << std::endl;
    assert(k.size(1) == page_size);
    std::cout << "Checking v.size(1) == " << page_size << ", got " << v.size(1) << std::endl;
    assert(v.size(1) == page_size);

    std::cout << "Checking k.size(2) == " << num_kv_heads << ", got " << k.size(2) << std::endl;
    assert(k.size(2) == num_kv_heads);
    std::cout << "Checking v.size(2) == " << num_kv_heads << ", got " << v.size(2) << std::endl;
    assert(v.size(2) == num_kv_heads);

    std::cout << "Checking k.size(3) == " << head_dim << ", got " << k.size(3) << std::endl;
    assert(k.size(3) == head_dim);
    std::cout << "Checking v.size(3) == " << head_dim << ", got " << v.size(3) << std::endl;
    assert(v.size(3) == head_dim);

    std::cout << "Checking k.device().type() == " << static_cast<int>(device.type()) << ", got "
              << static_cast<int>(k.device().type()) << std::endl;
    assert(k.device().type() == device.type());
    std::cout << "Checking v.device().type() == " << static_cast<int>(device.type()) << ", got "
              << static_cast<int>(v.device().type()) << std::endl;
    assert(v.device().type() == device.type());
  }

  std::cout << "--- TestConstructorCreatesCorrectShapeLayerFirst passed ---\n";
}

//------------------------------------------------------------------------------
// 测试不同层的 KCache / VCache 是否相互独立（地址不同）
//------------------------------------------------------------------------------
void TestKCacheAndVCacheAreIndependent() {
  std::cout << "\n=== TestKCacheAndVCacheAreIndependent ===" << std::endl;

  int num_layers = 2;
  int num_pages = 4;
  int num_kv_heads = 8;
  int head_dim = 64;
  torch::Device device = torch::kCPU;
  if (torch::cuda::is_available()) device = torch::kCUDA;
  torch::Dtype dtype = torch::kFloat32;

  MHAKVCache cache(num_layers, num_pages, num_kv_heads, head_dim, KVCacheLayout::kPageFirst, device, dtype);

  auto k0 = cache.KCache(0);
  auto k1 = cache.KCache(1);
  auto v0 = cache.VCache(0);
  auto v1 = cache.VCache(1);

  std::cout << "k0 data_ptr: " << k0.data_ptr() << ", k1 data_ptr: " << k1.data_ptr() << std::endl;
  std::cout << "v0 data_ptr: " << v0.data_ptr() << ", v1 data_ptr: " << v1.data_ptr() << std::endl;

  std::cout << "Checking k0.data_ptr() != k1.data_ptr()" << std::endl;
  assert(k0.data_ptr() != k1.data_ptr());
  std::cout << "Checking v0.data_ptr() != v1.data_ptr()" << std::endl;
  assert(v0.data_ptr() != v1.data_ptr());
  // 确保 K 和 V 也是独立的
  std::cout << "Checking k0.data_ptr() != v0.data_ptr()" << std::endl;
  assert(k0.data_ptr() != v0.data_ptr());

  std::cout << "--- TestKCacheAndVCacheAreIndependent passed ---\n";
}

//------------------------------------------------------------------------------
// 测试 StoreKV 的基本存储功能：写入后能正确读取
//------------------------------------------------------------------------------
void TestStoreKVStoresCorrectly() {
  std::cout << "\n=== TestStoreKVStoresCorrectly ===" << std::endl;

  int num_layers = 2;
  int num_pages = 8;
  int num_kv_heads = 8;
  int head_dim = 64;
  torch::Device device = torch::kCPU;
  if (torch::cuda::is_available()) device = torch::kCUDA;
  torch::Dtype dtype = torch::kFloat32;

  MHAKVCache cache(num_layers, num_pages, num_kv_heads, head_dim, KVCacheLayout::kPageFirst, device, dtype);

  int layer_id = 0;
  int batch_size = 2;
  int seq_len = 3;

  auto k =
      torch::randn({batch_size, seq_len, num_kv_heads, head_dim}, torch::TensorOptions().dtype(dtype).device(device));
  auto v =
      torch::randn({batch_size, seq_len, num_kv_heads, head_dim}, torch::TensorOptions().dtype(dtype).device(device));
  auto loc = RandomLoc(batch_size, seq_len, num_pages, device);
  std::cout << " loc : " << loc << std::endl;
  cache.StoreKV(k, v, loc, layer_id);

  auto k_cache = cache.KCache(layer_id);  // shape: [num_pages, 1, num_kv_heads, head_dim]
  auto v_cache = cache.VCache(layer_id);

  for (int b = 0; b < batch_size; ++b) {
    for (int s = 0; s < seq_len; ++s) {
      int page = loc[b * seq_len + s].item<int64_t>();
      auto k_input = k[b][s];         // [num_kv_heads, head_dim]
      auto v_input = v[b][s];         // [num_kv_heads, head_dim]
      auto k_cached = k_cache[page];  // [1, num_kv_heads, head_dim]
      auto v_cached = v_cache[page];  // [1, num_kv_heads, head_dim]

      // 去掉 page_size 维度（因为 page_size=1）
      k_cached = k_cached.squeeze(1);
      v_cached = v_cached.squeeze(1);

      bool k_ok = torch::allclose(k_input, k_cached);
      bool v_ok = torch::allclose(v_input, v_cached);

      std::cout << "Batch " << b << ", Seq " << s << ", page " << page << " -> K1 : " << k_input << std::endl;
      std::cout << "Batch " << b << ", Seq " << s << ", page " << page << " -> K2 : " << k_cached << std::endl;

      std::cout << "Batch " << b << ", Seq " << s << ", page " << page << " -> K allclose: " << (k_ok ? "yes" : "NO")
                << ", V allclose: " << (v_ok ? "yes" : "NO") << std::endl;

      std::cout << "Checking K allclose for batch " << b << " seq " << s << std::endl;
      assert(k_ok);
      std::cout << "Checking V allclose for batch " << b << " seq " << s << std::endl;
      assert(v_ok);
    }
  }

  std::cout << "--- TestStoreKVStoresCorrectly passed ---\n";
}

//------------------------------------------------------------------------------
// 测试多次写入同一 page 时的覆盖行为（后写入应覆盖前写入）
//------------------------------------------------------------------------------
void TestStoreKVOverwritesSamePage() {
  std::cout << "\n=== TestStoreKVOverwritesSamePage ===" << std::endl;

  int num_layers = 1;
  int num_pages = 2;
  int num_kv_heads = 4;
  int head_dim = 32;
  torch::Device device = torch::kCPU;
  if (torch::cuda::is_available()) device = torch::kCUDA;
  torch::Dtype dtype = torch::kFloat32;

  MHAKVCache cache(num_layers, num_pages, num_kv_heads, head_dim, KVCacheLayout::kPageFirst, device, dtype);

  int layer_id = 0;
  int target_page = 1;  // 我们将反复写入这个 page

  // 第一次写入
  auto k1 = torch::randn({1, 1, num_kv_heads, head_dim}, device).to(dtype);
  auto v1 = torch::randn({1, 1, num_kv_heads, head_dim}, device).to(dtype);
  auto loc1 = torch::full({1, 1}, target_page, torch::kLong).to(device);
  cache.StoreKV(k1, v1, loc1, layer_id);

  // 第二次写入（不同值）
  auto k2 = torch::randn({1, 1, num_kv_heads, head_dim}, device).to(dtype);
  auto v2 = torch::randn({1, 1, num_kv_heads, head_dim}, device).to(dtype);
  auto loc2 = torch::full({1, 1}, target_page, torch::kLong).to(device);
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
  std::cout << "After second write, K matches second: " << (k_match ? "yes" : "NO") << std::endl;
  std::cout << "After second write, V matches second: " << (v_match ? "yes" : "NO") << std::endl;

  std::cout << "Checking that K matches second write" << std::endl;
  assert(k_match);
  std::cout << "Checking that V matches second write" << std::endl;
  assert(v_match);

  std::cout << "--- TestStoreKVOverwritesSamePage passed ---\n";
}

//------------------------------------------------------------------------------
// 测试边界情况：空的输入（batch=0 或 seq=0）不应崩溃
//------------------------------------------------------------------------------
void TestStoreKVWithEmptyInput() {
  std::cout << "\n=== TestStoreKVWithEmptyInput ===" << std::endl;

  int num_layers = 2;
  int num_pages = 4;
  int num_kv_heads = 8;
  int head_dim = 64;
  torch::Device device = torch::kCPU;
  if (torch::cuda::is_available()) device = torch::kCUDA;
  torch::Dtype dtype = torch::kFloat32;

  MHAKVCache cache(num_layers, num_pages, num_kv_heads, head_dim, KVCacheLayout::kPageFirst, device, dtype);

  int layer_id = 0;

  // batch = 0
  auto k_empty_batch = torch::randn({0, 3, num_kv_heads, head_dim}, device).to(dtype);
  auto v_empty_batch = torch::randn({0, 3, num_kv_heads, head_dim}, device).to(dtype);
  auto loc_empty_batch = RandomLoc(0, 3, num_pages, device);
  std::cout << "Calling StoreKV with batch=0 (should not crash)" << std::endl;
  cache.StoreKV(k_empty_batch, v_empty_batch, loc_empty_batch, layer_id);  // 不应崩溃

  // seq = 0
  auto k_empty_seq = torch::randn({2, 0, num_kv_heads, head_dim}, device).to(dtype);
  auto v_empty_seq = torch::randn({2, 0, num_kv_heads, head_dim}, device).to(dtype);
  auto loc_empty_seq = RandomLoc(2, 0, num_pages, device);
  std::cout << "Calling StoreKV with seq=0 (should not crash)" << std::endl;
  cache.StoreKV(k_empty_seq, v_empty_seq, loc_empty_seq, layer_id);  // 不应崩溃

  std::cout << "Empty input handled without crash." << std::endl;
  std::cout << "--- TestStoreKVWithEmptyInput passed ---\n";
}

//------------------------------------------------------------------------------
// 测试 NumLayers 和 Device 接口返回正确值
//------------------------------------------------------------------------------
void TestNumLayersAndDevice() {
  std::cout << "\n=== TestNumLayersAndDevice ===" << std::endl;

  int num_layers = 3;
  int num_pages = 4;
  int num_kv_heads = 8;
  int head_dim = 64;
  torch::Device device = torch::kCPU;
  if (torch::cuda::is_available()) device = torch::kCUDA;
  torch::Dtype dtype = torch::kFloat32;

  MHAKVCache cache(num_layers, num_pages, num_kv_heads, head_dim, KVCacheLayout::kPageFirst, device, dtype);

  std::cout << "NumLayers(): " << cache.NumLayers() << " (expected " << num_layers << ")" << std::endl;
  std::cout << "Checking NumLayers() == " << num_layers << std::endl;
  assert(cache.NumLayers() == num_layers);

  std::cout << "Device().type(): " << static_cast<int>(cache.Device().type()) << " (expected "
            << static_cast<int>(device.type()) << ")" << std::endl;
  std::cout << "Checking Device().type() == " << static_cast<int>(device.type()) << std::endl;
  assert(cache.Device().type() == device.type());

  std::cout << "--- TestNumLayersAndDevice passed ---\n";
}

}  // namespace test
}  // namespace yllang

//------------------------------------------------------------------------------
// 主函数：运行所有测试
//------------------------------------------------------------------------------
int main() {
  // 注意：如果 CUDA 可用，所有测试将在 GPU 上运行；否则在 CPU 上运行。
  std::cout << "Running MHAKVCache unit tests...\n";

  yllang::test::TestConstructorCreatesCorrectShapeLayerFirst();
  yllang::test::TestConstructorCreatesCorrectShapePageFirst();
  yllang::test::TestKCacheAndVCacheAreIndependent();
  yllang::test::TestStoreKVStoresCorrectly();
  yllang::test::TestStoreKVOverwritesSamePage();
  yllang::test::TestStoreKVWithEmptyInput();
  yllang::test::TestNumLayersAndDevice();

  std::cout << "\nAll tests passed!\n";
  return 0;
}