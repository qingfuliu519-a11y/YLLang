#include "models/model_config.h"
#include "util/tensor.h"
namespace yllang {

ModelConfig::ModelConfig(const std::string &filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open config file: " + filename);
  }

  nlohmann::json j;
  file >> j;

  // Parse each field with appropriate type handling
  m_architectures_ = j.value("architectures", std::vector<std::string>{});
  m_attention_bias_ = j.value("attention_bias", false);
  m_attention_dropout_ = j.value("attention_dropout", 0.0);
  m_bos_token_id_ = j.value("bos_token_id", 0);
  m_eos_token_id_ = j.value("eos_token_id", 0);
  m_head_dim_ = j.value("head_dim", 0);
  m_hidden_act_ = j.value("hidden_act", "");
  m_hidden_size_ = j.value("hidden_size", 0);
  m_initializer_range_ = j.value("initializer_range", 0.0);
  m_intermediate_size_ = j.value("intermediate_size", 0);
  m_max_position_embeddings_ = j.value("max_position_embeddings", 0);
  m_max_window_layers_ = j.value("max_window_layers", 0);
  m_model_type_ = j.value("model_type", "");
  m_num_attention_heads_ = j.value("num_attention_heads", 0);
  m_num_hidden_layers_ = j.value("num_hidden_layers", 0);
  m_num_key_value_heads_ = j.value("num_key_value_heads", 0);
  m_rms_norm_eps_ = j.value("rms_norm_eps", 0.0);
  // rope_scaling can be null or object; here we treat as string, default empty
  if (j.contains("rope_scaling") && !j["rope_scaling"].is_null()) {
    m_rope_scaling_ = j["rope_scaling"].dump();
  } else {
    m_rope_scaling_ = "";
  }
  m_rope_theta_ = j.value("rope_theta", 0.0);
  if (j.contains("sliding_window") && !j["sliding_window"].is_null()) {
    m_sliding_window_ = j["sliding_window"].dump();
  } else {
    m_sliding_window_ = "";
  }
  m_tie_word_embeddings_ = j.value("tie_word_embeddings", false);
  m_torch_dtype_str_ = j.value("torch_dtype", "");
  m_torch_dtype_ = util::ParseTorchDtype(m_torch_dtype_str_);
  m_transformers_version_ = j.value("transformers_version", "");
  m_use_cache_ = j.value("use_cache", false);
  m_use_sliding_window_ = j.value("use_sliding_window", false);
  m_vocab_size_ = j.value("vocab_size", 0);
}
}  // namespace yllang