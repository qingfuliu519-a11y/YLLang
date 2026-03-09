//
// Created by lqf on 2026/2/28.
//

#ifndef YLLANG_LAYERS_BASE_LAYER_H
#define YLLANG_LAYERS_BASE_LAYER_H

#include <torch/torch.h>
namespace yllang {

class BaseLayer {
 public:
  BaseLayer() = default;
  virtual ~BaseLayer() = default;
  virtual auto Forward(const torch::Tensor &tensor) -> torch::Tensor = 0;
};

}  // namespace yllang

#endif  // YLLANG_LAYERS_BASE_LAYER_H
