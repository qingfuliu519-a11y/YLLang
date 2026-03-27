/**
 * @file linear.h
 * @brief Defines a linear (fully connected) layer.
 */

#ifndef YLLANG_LAYERS_LINEAR_LAYER_H
#define YLLANG_LAYERS_LINEAR_LAYER_H

#include <torch/torch.h>
#include "layers/base_layer.h"
#include "util/tensor.h"
namespace yllang {

/**
 * @brief A linear layer wrapping torch::nn::Linear.
 *
 * Provides a simple forward pass that applies the linear transformation.
 * Copy operations are deleted to avoid accidental duplication of parameters.
 */
class LinearLayer : public BaseLayer {
 public:
  // Deleted copy operations to enforce unique ownership of parameters.
  LinearLayer(const LinearLayer &) = delete;
  auto operator=(const LinearLayer &) -> LinearLayer & = delete;

  /**
   * @brief Constructs a LinearLayer.
   *
   * @param input_dim  Number of input features.
   * @param output_dim Number of output features.
   * @param bias       Whether to include a bias term.
   */
  LinearLayer(int input_dim, int output_dim, bool bias, torch::ScalarType dtype)
      : m_wight_(torch::nn::LinearOptions(input_dim, output_dim).bias(bias)) {
    m_wight_->weight = m_wight_->weight.toType(dtype);
    m_wight_->bias = m_wight_->bias.toType(dtype);
  }

  ~LinearLayer() override = default;

  auto SetWeights(torch::Tensor weights) -> void { throw std::runtime_error("should not come here"); }

  auto SetWeights(WeightLoader &loader) -> void override {
    util::CopyTensorWithCheck(m_wight_->weight, loader.Weights());
    loader.CompleteLayerLoad();
    util::CopyTensorWithCheck(m_wight_->bias, loader.Weights());
    loader.CompleteLayerLoad();
  }
  /**
   * @brief Forward pass: applies the linear transformation.
   *
   * @param tensor Input tensor of shape (..., input_dim).
   * @return torch::Tensor Output tensor of shape (..., output_dim).
   */
  auto Forward(const torch::Tensor &tensor) -> torch::Tensor override { return m_wight_(tensor); }

 protected:
  torch::nn::Linear m_wight_;  ///< Underlying linear module.
};

class QKVLinearLayer : public LinearLayer {
 public:
  QKVLinearLayer(int hidden_size, int num_qo_heads, int num_kv_heads, int head_dim, bool bias, torch::ScalarType dtype)
      : LinearLayer(hidden_size, (num_qo_heads + 2 * num_kv_heads) * head_dim, bias, dtype) {}

  auto SetWeights(torch::Tensor weights) -> void { throw std::runtime_error("should not come here"); }

  auto SetWeights(WeightLoader &loader) -> void override {
    util::CopyTensorWithCheck(m_wight_->weight, loader.Weights());
    loader.CompleteLayerLoad();
    util::CopyTensorWithCheck(m_wight_->bias, loader.Weights());
    loader.CompleteLayerLoad();
  }
};

}  // namespace yllang

#endif  // YLLANG_LAYERS_LINEAR_LAYER_H