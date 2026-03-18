#ifndef YLLANG_TOKENIZER_H
#define YLLANG_TOKENIZER_H
#include <memory>
#include "tokenizers_cpp.h"
namespace yllang {

class Tokenizer {
 public:
  Tokenizer() = default;

  ~Tokenizer() = default;

  auto FromBlobByteLevelBPE(const std::string &vocab_blob, const std::string &merges_blob,
                            const std::string &added_tokens = "") -> void {
    m_tokenizer_ = tokenizers::Tokenizer::FromBlobByteLevelBPE(vocab_blob, merges_blob, added_tokens);
  }

  auto FromBlobSentencePiece(const std::string &model_blob) -> void {
    m_tokenizer_ = tokenizers::Tokenizer::FromBlobSentencePiece(model_blob);
  }

  auto FromBlobRWKVWorld(const std::string &model_blob) -> void {
    m_tokenizer_ = tokenizers::Tokenizer::FromBlobSentencePiece(model_blob);
  }

  auto Encode(const std::string &text) const -> std::vector<int32_t> { return m_tokenizer_->Encode(text); }

  auto EncodeBatch(const std::vector<std::string> &texts) const -> std::vector<std::vector<int32_t>> {
    return m_tokenizer_->EncodeBatch(texts);
  }

  auto Decode(const std::vector<int32_t> &ids) const -> std::string { return m_tokenizer_->Decode(ids); }

  auto DecodeBatch(const std::vector<std::vector<int32_t>> &ids) const -> std::vector<std::string> {
    std::vector<std::string> texts;
    texts.reserve(ids.size());
    for (const auto &id : ids) {
      texts.push_back(Decode(id));
    }
    return texts;
  }

  auto Empty() const -> bool { return nullptr == m_tokenizer_; }

 private:
  std::unique_ptr<tokenizers::Tokenizer> m_tokenizer_{nullptr};
};

}  // namespace yllang

#endif  // YLLANG_TOKENIZER_H
