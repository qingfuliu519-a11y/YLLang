#ifndef YLLANG_LAYERS_RMS_NORM_H
#define YLLANG_LAYERS_RMS_NORM_H

#include <torch/torch.h>

namespace yllang {

class RMSNorm : BaseLayer {
 public:
  RMSNorm() = default;

  RMSNorm(int input_dim, float eps) : m_weight_(torch::empty({input_dim}, torch::TensorOptions())), m_eps_(eps) {}

  ~RMSNorm() override = default;

  auto Forward(const torch::Tensor &input) -> torch::Tensor override {
    return at::rms_norm(input, {input.size(-1)}, m_weight_, m_eps_);
  }

  RMSNorm(const RMSNorm &other) : m_weight_(other.m_weight_.clone()), m_eps_(other.m_eps_) {}

  RMSNorm(RMSNorm &&other) noexcept : m_weight_(std::move(other.m_weight_)), m_eps_(other.m_eps_) {}

  auto operator=(const RMSNorm &other) -> RMSNorm & {
    m_weight_ = other.m_weight_.clone();
    m_eps_ = other.m_eps_;
    return *this;
  }

  auto operator=(RMSNorm &&other) noexcept -> RMSNorm & {
    m_weight_ = std::move(other.m_weight_);
    m_eps_ = other.m_eps_;
    return *this;
  }

  auto Defined() const noexcept -> bool { return m_weight_.defined(); }

 private:
  torch::Tensor m_weight_;
  double m_eps_{0.0};
};

}  // namespace yllang

#endif  // YLLANG_LAYERS_RMS_NORM_H