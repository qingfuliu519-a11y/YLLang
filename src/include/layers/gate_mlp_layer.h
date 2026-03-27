#ifndef YLLANG_LAYERS_GATE_MLP_LAYER_H
#define YLLANG_LAYERS_GATE_MLP_LAYER_H

#include <torch/torch.h>
#include <memory>
#include <stdexcept>
#include "layers/base_layer.h"
#include "layers/linear.h"
#include "layers/weight_loader.h"
namespace yllang {

void SiluAndMul(torch::Tensor gate,torch::Tensor up,torch::Tensor output);

class GatedMLPLayer : public BaseLayer {
 public:
 using BaseLayer::SetWeights;
  GatedMLPLayer(int hidden_size, int intermediate_size, torch::ScalarType dtype)
      : m_gate_up_proj_(std::make_unique<LinearLayer>(hidden_size, 2 * intermediate_size, false,dtype)),
        m_down_proj_(std::make_unique<LinearLayer>(intermediate_size, hidden_size,false, dtype)) {}
  /**
   * @brief Performs the forward pass of the layer.
   *
   * @param tensor Input tensor.
   * @return torch::Tensor Output tensor after layer computation.
   */
  auto Forward(const torch::Tensor &input) -> torch::Tensor override {
    auto x  = m_gate_up_proj_->Forward(input);

    int64_t intermediate_size = x.size(1)/2;
    auto gate = x.narrow(1, 0, intermediate_size);
    auto up = x.narrow(1, intermediate_size, intermediate_size);
    auto output = torch::empty_like(gate);
    SiluAndMul(gate,up,output);
    return m_down_proj_->Forward(output);
  }

  auto SetWeights(WeightLoader &loader) -> void override {
    m_down_proj_->SetWeights(loader.Weights());
    loader.CompleteLayerLoad();
    auto gate_weight = loader.Weights();
    loader.CompleteLayerLoad();
    auto up_weight = loader.Weights();
    loader.CompleteLayerLoad();
    m_gate_up_proj_->SetWeights(torch::cat({gate_weight, up_weight}, 0));
  }

 private:
  std::unique_ptr<LinearLayer> m_gate_up_proj_;
  std::unique_ptr<LinearLayer> m_down_proj_;
};
}  // namespace yllang

#endif