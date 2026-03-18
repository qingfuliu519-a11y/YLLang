#include "models/model.h"
#include <filesystem>
#include "config/config.h"
namespace yllang {

struct ModelUrl {
  std::string m_tokenizer_json_;
  std::string m_vocab_json_;
  std::string m_model_safetensors_;
  std::string m_generation_config_json_;
  std::string m_config_json_;
};

std::string_view BaseUrlForType(ModelType type) {
  switch (type) {
    case ModelType::KQwen306b:
      return "https://huggingface.co/Qwen/Qwen3-0.6B-Base/resolve/main/";
  }
  return "";
}

ModelUrl ModelUrlByType(ModelType type) {
  auto base = BaseUrlForType(type);
  return ModelUrl{.m_tokenizer_json_ = std::string(base) + "tokenizer.json",
                  .m_vocab_json_ = std::string(base) + "vocab.json",
                  .m_model_safetensors_ = std::string(base) + "model.safetensors",
                  .m_generation_config_json_ = std::string(base) + "generation_config.json",
                  .m_config_json_ = std::string(base) + "config.json"};
}

auto Model::Load(ModelType model_type) -> std::unique_ptr<Model> {
  const std::string model_path = kModelPath + "/" + ModelTypeToString(model_type);

  std::filesystem::path dir(model_path);
  if (!std::filesystem::exists(dir)) {
    if (!std::filesystem::create_directories(dir)) {
      return nullptr;
    }
  }

  auto model_url = ModelUrlByType(model_type);

  return nullptr;
}

}  // namespace yllang
