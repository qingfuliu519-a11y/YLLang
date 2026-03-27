#ifndef YLLANG_MODEL_WEIGHT_CONTEXT_H
#define YLLANG_MODEL_WEIGHT_CONTEXT_H
#include <torch/torch.h>
#include <string>
#include "layers/base_layer.h"
#include "safetensors.hh"
#include "util/panic.h"
#include "util/tensor.h"
namespace yllang {
class WeightLoader {
 public:
  WeightLoader(const std::string &model_path) {
    std::string warn;
    std::string err;
#if defined(USE_MMAP)
    printf("USE mmap\n");
    bool ret = safetensors::mmap_from_file(model_path, &m_st_, &warn, &err);
#else
    bool ret = safetensors::load_from_file(model_path, &m_st_, &warn, &err);
#endif

    yllang::RuntimeCheck(warn.empty(), warn);
    yllang::RuntimeCheck(ret, "");
    yllang::RuntimeCheck(safetensors::validate_data_offsets(m_st_, err), err);
  }

  auto Weights() const -> torch::Tensor {
    assert(m_st_.tensors.size() > m_current_layer_index_);

    safetensors::tensor_t safe_tensor;
    m_st_.tensors.at(m_current_layer_index_, &safe_tensor);

    const void *data_ptr{nullptr};
    if (m_st_.mmaped) {
      data_ptr = static_cast<const void *>(m_st_.databuffer_addr);
    } else {
      data_ptr = static_cast<const void *>(m_st_.storage.data());
    }

    auto torch_type = util::SafetensorsType2TorchType(safe_tensor.dtype);
    auto shape_vec = std::vector<int64_t>{safe_tensor.shape.begin(), safe_tensor.shape.end()};
    auto torch_shape = torch::IntArrayRef{shape_vec};

    return torch::from_blob(const_cast<void *>(data_ptr), torch_shape, torch_type);
  }

  auto CurrentLayerIndex() const -> int { return m_current_layer_index_; }

  auto CompleteLayerLoad() -> void { ++m_current_layer_index_; }

 private:
  safetensors::safetensors_t m_st_;
  int m_current_layer_index_{0};
};
}  // namespace yllang
#endif