#ifndef YLLANG_MODEL_QWEN3_H
#define YLLANG_MODEL_QWEN3_H

#include <memory>
#include "models/model.h"
#include "models/model_loader.h"

namespace yllang {

class Qwen3 : public Model {
 public:
  Qwen3() = default;

  ~Qwen3() override = default;

  /**
   * @brief Returns the list of files to download for a given model type.
   *
   * The returned list contains pairs of remote URL and local filename (relative
   * to the model directory). The URLs are constructed using a mirror base URL.
   *
   * @param type The model type (e.g., KQwen306b).
   * @return std::vector<ModelFileEntry> Vector of file entries. Empty if the model type is unknown.
   */
  static auto ModelFiles() -> std::vector<ModelFileEntry>;

  static auto Kind() -> ModelType { return ModelType::KQwen306b; }

  static auto Load() -> std::unique_ptr<Model>;

  auto Forward(const std::shared_ptr<yllang::Batch> &batch) -> torch::Tensor override { return {}; }

 private:
  inline static RegisterHelper<Qwen3> m_register_helper_;
};

}  // namespace yllang

#endif  // YLLANG_MODEL_QWEN3_H
