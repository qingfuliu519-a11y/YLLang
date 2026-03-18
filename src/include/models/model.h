#ifndef YLLANG_MODEL_H
#define YLLANG_MODEL_H

#include <torch/torch.h>
#include <map>
#include <memory>
#include <string>
#include "batch/batch.h"
namespace yllang {

enum class ModelType : std::uint8_t { KQwen306b = 0 };

inline auto ModelTypeToString(ModelType model_type) noexcept -> std::string {
  switch (model_type) {
#define MODELTYPETOSTRING(X) \
  case ModelType::X:         \
    return #X;

    MODELTYPETOSTRING(KQwen306b)

#undef MODELTYPETOSTRING
    default:
      break;
  }
  return "None";
}

class Model {
 public:
  static auto Load(ModelType model_type) -> std::unique_ptr<Model>;

  auto Forward(const std::shared_ptr<yllang::Batch> &batch) -> torch::Tensor;
};
}  // namespace yllang

#endif  // YLLANG_MODEL_H
