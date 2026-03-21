/**
 * @file batch.h
 * @brief Defines the Batch class that encapsulates a set of requests for processing.
 */

#ifndef YLLANG_BATCH_H
#define YLLANG_BATCH_H

#include <torch/torch.h>
#include <vector>
#include "request/request.h"

namespace yllang {

/**
 * @brief Represents a batch of requests ready for inference.
 *
 * Contains the concatenated input IDs, output locations, and index tensors
 * for loading from and writing to the KV cache.
 */
class Batch {
 public:
  Batch() = default;
  /**
   * @brief Constructs a Batch from request list and associated tensors.
   *
   * @param request_list   List of request objects.
   * @param input_ids      Concatenated input token IDs for the batch.
   * @param out_loc        Output location indices (page indices for new tokens).
   * @param load_indices   Indices for loading cached data.
   * @param write_indices  Indices for writing new data to cache.
   */
  Batch(std::vector<std::shared_ptr<Request>> request_list, torch::Tensor input_ids, torch::Tensor out_loc,
        torch::Tensor load_indices, torch::Tensor write_indices)
      : m_request_(std::move(request_list)),
        m_input_ids_(input_ids),
        m_out_loc_(std::move(out_loc)),
        m_load_indices_(std::move(load_indices)),
        m_write_indices_(std::move(write_indices)) {}

  auto InputIds() -> torch::Tensor & { return m_input_ids_; }

  auto Requests() const -> const std::vector<std::shared_ptr<Request>> & { return m_request_; }

 private:
  std::vector<std::shared_ptr<Request>> m_request_;  ///< List of requests in the batch.
  torch::Tensor m_input_ids_;                        ///< 1D tensor of concatenated input token IDs.
  torch::Tensor m_out_loc_;                          ///< 1D tensor of output page indices.
  torch::Tensor m_load_indices_;                     ///< 1D tensor of indices for loading from cache.
  torch::Tensor m_write_indices_;                    ///< 1D tensor of indices for writing to cache.
};

}  // namespace yllang

#endif  // YLLANG_BATCH_H