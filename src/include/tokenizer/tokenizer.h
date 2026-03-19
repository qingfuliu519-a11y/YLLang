/**
 * @file tokenizer.h
 * @brief Defines a wrapper class for the tokenizers_cpp library.
 */

#ifndef YLLANG_TOKENIZER_H
#define YLLANG_TOKENIZER_H

#include <memory>
#include <string>
#include "batch/batch.h"
#include "tokenizers_cpp.h"

namespace yllang {

/**
 * @brief Wrapper around tokenizers_cpp::Tokenizer for easy integration.
 *
 * This class provides a unified interface for loading different tokenizer types
 * (ByteLevelBPE, SentencePiece, RWKVWorld, JSON) and performing encoding/decoding.
 * It manages the underlying tokenizer instance as a unique pointer.
 */
class Tokenizer {
 public:
  Tokenizer() = default;
  ~Tokenizer() = default;

  // -------------------------------------------------------------------------
  // Initialization methods (re‑initialize an existing Tokenizer)
  // -------------------------------------------------------------------------

  /**
   * @brief Initializes the tokenizer from in-memory ByteLevelBPE data.
   *
   * @param vocab_blob   Byte string containing the vocabulary.
   * @param merges_blob  Byte string containing the BPE merges.
   * @param added_tokens Optional byte string for added tokens (default empty).
   */
  auto FromBlobByteLevelBPE(const std::string &vocab_blob, const std::string &merges_blob,
                            const std::string &added_tokens = "") -> void {
    m_tokenizer_ = tokenizers::Tokenizer::FromBlobByteLevelBPE(vocab_blob, merges_blob, added_tokens);
  }

  /**
   * @brief Initializes the tokenizer from a SentencePiece model blob.
   *
   * @param model_blob Byte string containing the SentencePiece model.
   */
  auto FromBlobSentencePiece(const std::string &model_blob) -> void {
    m_tokenizer_ = tokenizers::Tokenizer::FromBlobSentencePiece(model_blob);
  }

  /**
   * @brief Initializes the tokenizer from an RWKV World model blob.
   *
   * Currently delegates to FromBlobSentencePiece.
   *
   * @param model_blob Byte string containing the RWKV World model.
   */
  auto FromBlobRWKVWorld(const std::string &model_blob) -> void {
    m_tokenizer_ = tokenizers::Tokenizer::FromBlobSentencePiece(model_blob);
  }

  /**
   * @brief Initializes the tokenizer from a JSON‑formatted tokenizer file.
   *
   * @param tokenizer_json Byte string containing the JSON tokenizer definition.
   */
  auto FromBlobJSON(const std::string &tokenizer_json) -> void {
    // Assuming tokenizers::Tokenizer::FromBlobJSON exists (implementation detail).
    m_tokenizer_ = tokenizers::Tokenizer::FromBlobJSON(tokenizer_json);
  }

  // -------------------------------------------------------------------------
  // Encoding and decoding
  // -------------------------------------------------------------------------

  /**
   * @brief Encodes a single text string into token IDs.
   *
   * @param text Input text.
   * @return std::vector<int32_t> Sequence of token IDs.
   */
  auto Encode(const std::string &text) const -> std::vector<int32_t> { return m_tokenizer_->Encode(text); }

  /**
   * @brief Encodes multiple text strings in batch.
   *
   * @param texts Vector of input texts.
   * @return std::vector<std::vector<int32_t>> Batch of token ID sequences.
   */
  auto EncodeBatch(const std::vector<std::string> &texts) const -> std::vector<std::vector<int32_t>> {
    return m_tokenizer_->EncodeBatch(texts);
  }

  /**
   * @brief Decodes token IDs back to text.
   *
   * @param ids Sequence of token IDs.
   * @return std::string Decoded text.
   */
  auto Decode(const std::vector<int32_t> &ids) const -> std::string { return m_tokenizer_->Decode(ids); }

  /**
   * @brief Decodes a batch of token ID sequences.
   *
   * @param ids Batch of token ID sequences.
   * @return std::vector<std::string> Decoded texts.
   */
  auto DecodeBatch(const std::vector<std::vector<int32_t>> &ids) const -> std::vector<std::string> {
    std::vector<std::string> texts;
    texts.reserve(ids.size());
    for (const auto &id : ids) {
      texts.push_back(Decode(id));
    }
    return texts;
  }

  /**
   * @brief Checks whether the tokenizer has been initialized.
   *
   * @return bool True if no tokenizer is loaded, false otherwise.
   */
  auto Empty() const -> bool { return nullptr == m_tokenizer_; }

 private:
  std::unique_ptr<tokenizers::Tokenizer> m_tokenizer_{nullptr};  ///< Underlying tokenizer instance.
};

// -------------------------------------------------------------------------
// Factory functions that return a fully constructed Tokenizer
// -------------------------------------------------------------------------

/**
 * @brief Creates a Tokenizer from ByteLevelBPE blob data.
 *
 * @param vocab_blob   Byte string containing the vocabulary.
 * @param merges_blob  Byte string containing the BPE merges.
 * @param added_tokens Optional byte string for added tokens (default empty).
 * @return std::unique_ptr<Tokenizer> A new Tokenizer instance.
 */
inline auto FromBlobByteLevelBPE(const std::string &vocab_blob, const std::string &merges_blob,
                                 const std::string &added_tokens = "") -> std::unique_ptr<Tokenizer> {
  auto tokenizer = std::make_unique<Tokenizer>();
  tokenizer->FromBlobByteLevelBPE(vocab_blob, merges_blob, added_tokens);
  return tokenizer;
}

/**
 * @brief Creates a Tokenizer from a SentencePiece model blob.
 *
 * @param model_blob Byte string containing the SentencePiece model.
 * @return std::unique_ptr<Tokenizer> A new Tokenizer instance.
 */
inline auto FromBlobSentencePiece(const std::string &model_blob) -> std::unique_ptr<Tokenizer> {
  auto tokenizer = std::make_unique<Tokenizer>();
  tokenizer->FromBlobSentencePiece(model_blob);
  return tokenizer;
}

/**
 * @brief Creates a Tokenizer from an RWKV World model blob.
 *
 * @param model_blob Byte string containing the RWKV World model.
 * @return std::unique_ptr<Tokenizer> A new Tokenizer instance.
 */
inline auto FromBlobRWKVWorld(const std::string &model_blob) -> std::unique_ptr<Tokenizer> {
  auto tokenizer = std::make_unique<Tokenizer>();
  tokenizer->FromBlobRWKVWorld(model_blob);
  return tokenizer;
}

/**
 * @brief Creates a Tokenizer from a JSON‑formatted tokenizer definition.
 *
 * @param tokenizer_json Byte string containing the JSON tokenizer definition.
 * @return std::unique_ptr<Tokenizer> A new Tokenizer instance.
 */
inline auto FromBlobJSON(const std::string &tokenizer_json) -> std::unique_ptr<Tokenizer> {
  auto tokenizer = std::make_unique<Tokenizer>();
  tokenizer->FromBlobJSON(tokenizer_json);
  return tokenizer;
}

}  // namespace yllang

#endif  // YLLANG_TOKENIZER_H