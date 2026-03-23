/**
 * @file model_config.h
 * @brief Defines the ModelConfig class for loading and accessing model configuration from JSON.
 */

#ifndef YLLANG_CONFIG_MODEL_CONFIG_H
#define YLLANG_CONFIG_MODEL_CONFIG_H

#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace yllang {

/**
 * @brief Loads and stores model configuration parameters from a JSON file.
 *
 * This class reads a Hugging Face style config.json and provides getters for
 * all standard fields (architectures, attention settings, hidden sizes, etc.).
 * The constructor parses the file and initializes all member variables.
 */
class ModelConfig {
 public:
  /**
   * @brief Constructs a ModelConfig by parsing the given JSON file.
   *
   * @param filename Path to the config.json file.
   * @throws std::runtime_error if file cannot be opened or JSON is invalid.
   */
  explicit ModelConfig(const std::string &filename);

  /// Returns the list of model architecture names (e.g., ["Qwen2ForCausalLM"]).
  auto Architectures() const -> const std::vector<std::string> & { return m_architectures_; }

  /// Returns whether attention layers use bias.
  auto AttentionBias() const -> bool { return m_attention_bias_; }

  /// Returns the dropout probability for attention layers.
  auto AttentionDropout() const -> double { return m_attention_dropout_; }

  /// Returns the token ID for beginning-of-sequence (BOS).
  auto BosTokenId() const -> int { return m_bos_token_id_; }

  /// Returns the token ID for end-of-sequence (EOS).
  auto EosTokenId() const -> int { return m_eos_token_id_; }

  /// Returns the dimension of each attention head.
  auto HeadDim() const -> int { return m_head_dim_; }

  /// Returns the activation function used in feed-forward layers.
  auto HiddenAct() const -> const std::string & { return m_hidden_act_; }

  /// Returns the hidden size (embedding dimension) of the model.
  auto HiddenSize() const -> int { return m_hidden_size_; }

  /// Returns the range for parameter initialization.
  auto InitializerRange() const -> double { return m_initializer_range_; }

  /// Returns the intermediate size of feed-forward layers.
  auto IntermediateSize() const -> int { return m_intermediate_size_; }

  /// Returns the maximum position embedding length (context window).
  auto MaxPositionEmbeddings() const -> int { return m_max_position_embeddings_; }

  /// Returns the number of layers that use sliding window attention (if any).
  auto MaxWindowLayers() const -> int { return m_max_window_layers_; }

  /// Returns the model type identifier (e.g., "qwen2").
  auto ModelType() const -> const std::string & { return m_model_type_; }

  /// Returns the number of query/output attention heads.
  auto NumAttentionHeads() const -> int { return m_num_attention_heads_; }

  /// Returns the number of hidden layers.
  auto NumHiddenLayers() const -> int { return m_num_hidden_layers_; }

  /// Returns the number of key/value heads (for GQA/MQA).
  auto NumKeyValueHeads() const -> int { return m_num_key_value_heads_; }

  /// Returns the epsilon for RMS normalization.
  auto RmsNormEps() const -> double { return m_rms_norm_eps_; }

  /// Returns the RoPE scaling configuration string.
  auto RopeScaling() const -> const std::string & { return m_rope_scaling_; }

  /// Returns the base theta for RoPE.
  auto RopeTheta() const -> double { return m_rope_theta_; }

  /// Returns the sliding window configuration string.
  auto SlidingWindow() const -> const std::string & { return m_sliding_window_; }

  /// Returns whether word embeddings are tied to output projections.
  auto TieWordEmbeddings() const -> bool { return m_tie_word_embeddings_; }

  /// Returns the torch data type (e.g., "float16", "bfloat16").
  auto TorchDtype() const -> const std::string & { return m_torch_dtype_; }

  /// Returns the transformers library version that generated this config.
  auto TransformersVersion() const -> const std::string & { return m_transformers_version_; }

  /// Returns whether the KV cache is enabled.
  auto UseCache() const -> bool { return m_use_cache_; }

  /// Returns whether sliding window attention is used.
  auto UseSlidingWindow() const -> bool { return m_use_sliding_window_; }

  /// Returns the vocabulary size.
  auto VocabSize() const -> int { return m_vocab_size_; }

 private:
  std::vector<std::string> m_architectures_;  ///< Model architecture names.
  bool m_attention_bias_;                     ///< Whether attention uses bias.
  double m_attention_dropout_;                ///< Attention dropout probability.
  int m_bos_token_id_;                        ///< Beginning-of-sequence token ID.
  int m_eos_token_id_;                        ///< End-of-sequence token ID.
  int m_head_dim_;                            ///< Dimension per attention head.
  std::string m_hidden_act_;                  ///< Hidden layer activation function.
  int m_hidden_size_;                         ///< Hidden layer dimension.
  double m_initializer_range_;                ///< Parameter initialization range.
  int m_intermediate_size_;                   ///< Feed-forward intermediate size.
  int m_max_position_embeddings_;             ///< Maximum context length.
  int m_max_window_layers_;                   ///< Number of sliding window layers.
  std::string m_model_type_;                  ///< Model type identifier.
  int m_num_attention_heads_;                 ///< Number of query/output heads.
  int m_num_hidden_layers_;                   ///< Number of transformer layers.
  int m_num_key_value_heads_;                 ///< Number of key/value heads.
  double m_rms_norm_eps_;                     ///< RMS norm epsilon.
  std::string m_rope_scaling_;                ///< RoPE scaling configuration.
  double m_rope_theta_;                       ///< RoPE base theta.
  std::string m_sliding_window_;              ///< Sliding window configuration.
  bool m_tie_word_embeddings_;                ///< Whether embeddings are tied.
  std::string m_torch_dtype_;                 ///< Torch data type.
  std::string m_transformers_version_;        ///< Transformers library version.
  bool m_use_cache_;                          ///< Whether KV cache is used.
  bool m_use_sliding_window_;                 ///< Whether sliding window is used.
  int m_vocab_size_;                          ///< Vocabulary size.
};

}  // namespace yllang

#endif  // YLLANG_CONFIG_MODEL_CONFIG_H