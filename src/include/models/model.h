#ifndef YLLANG_MODEL_H
#define YLLANG_MODEL_H

#include <torch/torch.h>
#include <memory>
#include <string>
#include <vector>
#include "batch/batch.h"
#include "tokenizer/tokenizer.h"

namespace yllang {

enum class ModelType : std::uint8_t { KNone = 0, KQwen306b = 0 };

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

/// Alias for a pair representing a model file: (remote URL, local relative filename).
using ModelFileEntry = std::pair<std::string, std::string>;

class Model {
 public:
  virtual ~Model() = default;

  /**
   * @brief Returns the list of files to download for a given model type.
   *
   * The returned list contains pairs of remote URL and local filename (relative
   * to the model directory). The URLs are constructed using a mirror base URL.
   *
   * @param type The model type (e.g., KQwen306b).
   * @return std::vector<ModelFileEntry> Vector of file entries. Empty if the model type is unknown.
   */
  static auto ModelFiles() -> std::vector<ModelFileEntry> { return {}; }

  static auto Kind() -> ModelType { return ModelType::KNone; }

  virtual auto Forward(const std::shared_ptr<yllang::Batch> &batch) -> torch::Tensor = 0;

 protected:
  auto Tokenize(const std::shared_ptr<yllang::Batch> &batch);

 protected:
};
}  // namespace yllang

#endif  // YLLANG_MODEL_H
