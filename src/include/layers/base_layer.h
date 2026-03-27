/**
 * @file base_layer.h
 * @brief Defines an abstract base class for all neural network layers.
 */

#ifndef YLLANG_LAYERS_BASE_LAYER_H
#define YLLANG_LAYERS_BASE_LAYER_H
#include <torch/torch.h>
#include <stdexcept>
#include "layers/weight_loader.h"

namespace yllang {

/**
 * @brief Abstract interface for a layer in the neural network.
 *
 * All concrete layers (e.g., Linear, Attention) must implement the Forward method.
 */
class BaseLayer {
 public:
  BaseLayer() = default;
  virtual ~BaseLayer() = default;

  /**
   * @brief Performs the forward pass of the layer.
   *
   * @param tensor Input tensor.
   * @return torch::Tensor Output tensor after layer computation.
   */
  virtual auto Forward(const torch::Tensor &tensor) -> torch::Tensor {
    throw std::runtime_error("should not come here");
  }

  virtual auto Forward(const torch::Tensor &tensor, const torch::Tensor &residual)
      -> std::pair<torch::Tensor, torch::Tensor> {
    throw std::runtime_error("should not come here");
  }

  virtual auto SetWeights(torch::Tensor weights) -> void = 0;

  virtual auto SetWeights(WeightLoader &loader) -> void {
    SetWeights(loader.Weights());
    loader.CompleteLayerLoad();
  }
};

}  // namespace yllang

#endif  // YLLANG_LAYERS_BASE_LAYER_H