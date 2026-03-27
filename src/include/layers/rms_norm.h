/**
 * @file rms_norm.h
 * @brief Defines the RMSNorm layer for root mean square layer normalization.
 */

#ifndef YLLANG_LAYERS_RMS_NORM_H
#define YLLANG_LAYERS_RMS_NORM_H

#include <torch/torch.h>
#include "layers/base_layer.h"
#include "util/tensor.h"
namespace yllang {

void RSMNormWithoutResidual(torch::Tensor input, torch::Tensor ouput, torch::Tensor weight, float eps);

void RSMNormWithResidual(torch::Tensor input, torch::Tensor residual, torch::Tensor weight, float eps);
/**
 * @brief Root Mean Square Normalization layer.
 *
 * Applies RMS normalization to the input tensor along the last dimension.
 * Inherits from BaseLayer (assumed to be defined elsewhere).
 */
class RMSNorm : public BaseLayer {
 public:
 using BaseLayer::SetWeights;
 
  RMSNorm() = default;

  /**
   * @brief Constructs an RMSNorm layer with given input dimension and epsilon.
   *
   * @param input_dim Dimensionality of the input (last dimension).
   * @param eps       Small epsilon to avoid division by zero.
   */
  RMSNorm(int input_dim, float eps, torch::ScalarType dtype)
      : m_weight_(torch::empty({input_dim}, torch::TensorOptions(dtype))), m_eps_(eps) {}

  RMSNorm(int input_dim, torch::ScalarType dtype)
      : m_weight_(torch::empty({input_dim}, torch::TensorOptions(dtype))), m_eps_(1e-5) {}

  ~RMSNorm() override = default;

  /**
   * @brief Forward pass: applies RMS normalization.
   *
   * @param input Input tensor of shape (..., input_dim).
   * @return torch::Tensor Normalized output.
   */
  auto Forward(const torch::Tensor &input) -> torch::Tensor override {
    auto output = torch::empty_like(input);
    RSMNormWithoutResidual(input, output, m_weight_, m_eps_);
    return output;
  }

  auto Forward(const torch::Tensor &input, const torch::Tensor &residual)
      -> std::pair<torch::Tensor, torch::Tensor> override {
    RSMNormWithResidual(input, residual, m_weight_, m_eps_);
    return {input, residual};
  }
  // Copy constructor
  RMSNorm(const RMSNorm &other) : m_weight_(other.m_weight_.clone()), m_eps_(other.m_eps_) {}

  // Move constructor
  RMSNorm(RMSNorm &&other) noexcept : m_weight_(std::move(other.m_weight_)), m_eps_(other.m_eps_) {}

  // Copy assignment
  auto operator=(const RMSNorm &other) -> RMSNorm & {
    m_weight_ = other.m_weight_.clone();
    m_eps_ = other.m_eps_;
    return *this;
  }

  // Move assignment
  auto operator=(RMSNorm &&other) noexcept -> RMSNorm & {
    m_weight_ = std::move(other.m_weight_);
    m_eps_ = other.m_eps_;
    return *this;
  }

  auto SetWeights(torch::Tensor weights) -> void override { util::CopyTensorWithCheck(m_weight_, weights); }

  /**
   * @brief Checks if the layer has been properly initialized (weight tensor defined).
   *
   * @return bool True if weight is defined.
   */
  auto Defined() const noexcept -> bool { return m_weight_.defined(); }

 private:
  torch::Tensor m_weight_;    ///< Learnable weight parameter.
  torch::Tensor m_residual_;  ///< Learnable weight parameter.
  double m_eps_{0.0};         ///< Epsilon for numerical stability.
};

}  // namespace yllang

#endif  // YLLANG_LAYERS_RMS_NORM_H