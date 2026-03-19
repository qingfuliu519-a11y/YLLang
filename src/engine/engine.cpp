#include "engine/engine.h"

namespace yllang {
auto Engine::Tokenize(const std::unique_ptr<UserMsg> &user_msg) const -> torch::Tensor {
  if (!m_tokenizer_ || m_tokenizer_->Empty()) {
    throw std::runtime_error("Tokenizer not initialized");
  }

  const std::string &text = user_msg->InputMsg();

  std::vector<int32_t> ids = m_tokenizer_->Encode(text);

  return torch::tensor(ids, torch::kInt32);
}
}  // namespace yllang
