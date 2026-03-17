#ifndef YLLANG_BATCH_H
#define YLLANG_BATCH_H

#include <torch/torch.h>
#include <vector>
#include "request/request.h"

namespace yllang {
class Batch {
 public:
  Batch(std::vector<std::shared_ptr<Request>> request_list, torch::Tensor input_ids, torch::Tensor out_loc,
        torch::Tensor load_indices, torch::Tensor write_indices)
      : m_request_(std::move(request_list)),
        m_input_ids_(std::move(input_ids)),
        m_out_loc_(std::move(out_loc)),
        m_load_indices_(std::move(load_indices)),
        m_write_indices_(std::move(write_indices)) {}

 private:
  std::vector<std::shared_ptr<Request>> m_request_;
  torch::Tensor m_input_ids_;
  torch::Tensor m_out_loc_;
  torch::Tensor m_load_indices_;
  torch::Tensor m_write_indices_;
};
}  // namespace yllang

#endif  // YLLANG_BATCH_H
