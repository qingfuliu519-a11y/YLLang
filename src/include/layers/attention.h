#ifndef YLLANG_LAYERS_ATTENTION_H
#define YLLANG_LAYERS_ATTENTION_H

#include <torch/torch.h>
#include "layers/base_layer.h"
#include "layers/rms_norm.h"
namespace yllang {

class Attention : public BaseLayer {
 public:
  Attention(int layer_id, int num_qo_heads, int num_kv_heads, int head_dim, RMSNorm qo_norm, RMSNorm k_norm)
      : m_layer_id_(layer_id),
        m_num_qo_heads_(num_qo_heads),
        m_num_kv_heads_(num_kv_heads),
        m_head_dim_(head_dim),
        m_qo_dim_(m_num_qo_heads_ * head_dim),
        m_kv_dim_(m_num_kv_heads_ * head_dim),
        m_qo_norm_(qo_norm),
        m_k_norm_(k_norm),
        m_mha(torch::nn::MultiheadAttentionOptions()) {}

  ~Attention() override = default;

  auto Forward(const torch::Tensor &qkv_raw) -> torch::Tensor override { return {}; }

 private:
  int m_layer_id_;
  int m_num_qo_heads_;
  int m_num_kv_heads_;
  int m_head_dim_;
  int m_qo_dim_;
  int m_kv_dim_;
  RMSNorm m_qo_norm_;
  RMSNorm m_k_norm_;
};

}  // namespace yllang

#endif  // YLLANG_LAYERS_ATTENTION_H