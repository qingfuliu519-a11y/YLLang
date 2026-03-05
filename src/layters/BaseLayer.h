//
// Created by lqf on 2026/2/28.
//

#ifndef BASELAYER_H
#define BASELAYER_H

#include <torch/torch.h>
class BaseLayer {
 public:
  BaseLayer() = default;
  virtual ~BaseLayer() = default;
  virtual auto Forward() -> torch::Tensor = 0;
};

#endif  // BASELAYER_H
