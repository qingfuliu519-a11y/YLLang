#ifndef YLLANG_ENGINE_H_
#define YLLANG_ENGINE_H_

#include <torch/torch.h>
#include "request/request.h"
#include "tokenizer/tokenizer.h"
namespace yllang {
class Engine {
 public:
 protected:
  auto Tokenize(const std::unique_ptr<UserMsg> &user_msg) const -> torch::Tensor;

 protected:
  std::unique_ptr<Tokenizer> m_tokenizer_;
};
}  // namespace yllang

#endif  // YLLANG_ENGINE_H_