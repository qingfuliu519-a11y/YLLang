#ifndef YLLANG_MODEL_WEIGHT_CONTEXT_H
#define YLLANG_MODEL_WEIGHT_CONTEXT_H
#include <string>
#include "safetensors.hh"
#include "util/panic.h"
namespace yllang {
class ModelWeightContext {
 public:
  ModelWeightContext(const std::string &model_path) {
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

  auto Weights() const -> const safetensors::safetensors_t & { return m_st_; }

  auto CurrentLayerIndex() const -> int { return m_current_layer_index_; }

  auto CompleteLayerLoad() -> void { ++m_current_layer_index_; }

 private:
  safetensors::safetensors_t m_st_;
  int m_current_layer_index_{0};
};
}  // namespace yllang
#endif