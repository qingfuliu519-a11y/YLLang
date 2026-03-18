/**
 * @file linear.h
 * @brief Defines a linear (fully connected) layer.
 */

#ifndef YLLANG_LAYERS_LINEAR_LAYER_H
#define YLLANG_LAYERS_LINEAR_LAYER_H

#include <torch/torch.h>
#include "layers/base_layer.h"

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
  LinearLayer(int input_dim, int output_dim, bool bias)
      : m_wight_(torch::nn::LinearOptions(input_dim, output_dim).bias(bias)) {}

  ~LinearLayer() override = default;

  /**
   * @brief Forward pass: applies the linear transformation.
   *
   * @param tensor Input tensor of shape (..., input_dim).
   * @return torch::Tensor Output tensor of shape (..., output_dim).
   */
  auto Forward(const torch::Tensor &tensor) -> torch::Tensor override { return m_wight_(tensor); }

 private:
  torch::nn::Linear m_wight_;  ///< Underlying linear module.
};

}  // namespace yllang

#endif  // YLLANG_LAYERS_LINEAR_LAYER_H