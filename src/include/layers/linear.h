//
// Created by lqf on 2026/2/28.
//

#ifndef YLLANG_LAYERS_LINEAR_LAYER_H
#define YLLANG_LAYERS_LINEAR_LAYER_H

#include <torch/torch.h>
#include "layers/base_layer.h"

namespace yllang {

class LinearLayer : public BaseLayer {
 public:
  LinearLayer(const LinearLayer &) = delete;

  auto operator=(const LinearLayer &) -> LinearLayer & = delete;

  LinearLayer(int input_dim, int output_dim, bool bias)
      : m_wight_(torch::nn::LinearOptions(input_dim, output_dim).bias(bias)) {}

  ~LinearLayer() override = default;

  auto Forward(const torch::Tensor &tensor) -> torch::Tensor override { return m_wight_(tensor); }

 private:
  torch::nn::Linear m_wight_;
};
}  // namespace yllang

#endif  // YLLANG_LAYERS_LINEAR_LAYER_H
