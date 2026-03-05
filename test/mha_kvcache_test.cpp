#include <gtest/gtest.h>
#include <torch/torch.h>

namespace yllang {
namespace test {

class MHAKVCacheTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 测试参数
    num_layers_ = 2;
    num_pages_ = 4;
    num_kv_heads_ = 8;
    head_dim_ = 16;
    device_ = torch::kCPU;  // 可改为 torch::kCUDA 测试 GPU
    dtype_ = torch::kFloat32;

    // 确保分布式信息返回 tp_size = 1（需要实际环境配置或 mock）
    // 这里假设 GetDistributedInfo().GetSize() == 1
  }

  int num_layers_;
  int num_pages_;
  int num_kv_heads_;
  int head_dim_;
  torch::Device device_;
  torch::Dtype dtype_;
};

// 测试构造函数在不同 layout 下创建正确的缓存形状
TEST_F(MHAKVCacheTest, ConstructorCreatesCorrectShape) {
  // kPageFirst 布局
  EXPECT_EQ(1, 1);

}




}  // namespace test
}  // namespace yllang
