/**
 * @file vocab_embedding.h
 * @brief Defines the vocabulary embedding layer that maps token IDs to dense vectors.
 */

#ifndef YLLANG_LAYERS_VOCAB_EMBEDDING_H
#define YLLANG_LAYERS_VOCAB_EMBEDDING_H

#include <torch/torch.h>
#include "config/config.h"
#include "layers/base_layer.h"
#include "layers/index.h"

namespace yllang {

/**
 * @brief Embedding layer that converts token indices into dense vector representations.
 *
 * This layer stores a learnable weight matrix of shape [vocab_size, embedding_dim].
 * During forward pass, it gathers rows corresponding to input indices using the
 * high‑performance Index kernel.
 */
class VocabEmbeddingLayer : public BaseLayer {
 public:
  /**
   * @brief Constructs a VocabEmbedding layer.
   *
   * @param vocab_size    Size of the vocabulary (number of distinct tokens).
   * @param embedding_dim Dimension of the output embedding vectors.
   */
  VocabEmbeddingLayer(int vocab_size, int embedding_dim, torch::ScalarType dtype)
      : m_weights_(torch::empty({vocab_size, embedding_dim}, dtype)), m_embedding_dim_(embedding_dim) {}

  /**
   * @brief Forward pass: retrieves embeddings for the given token indices.
   *
   * @param indices 1D tensor of token IDs.
   * @return torch::Tensor Output tensor of shape [indices.size(0), embedding_dim].
   */
  auto Forward(const torch::Tensor &indices) -> torch::Tensor override {
    TORCH_CHECK(1 == indices.dim(), "Expect a 1D tensor of token IDs");
    torch::Tensor output = torch::empty({indices.size(0), m_embedding_dim_});
    Index<kElementSize>(indices, m_weights_, output);
    return output;
  }

  auto SetWeights(torch::Tensor weights) -> void override {
    TORCH_CHECK(m_weights_.sizes() == weights.sizes(), "Tensor shape mismatch. Expected ", m_weights_.sizes(), ", got ",
                weights.sizes());
    TORCH_CHECK(m_weights_.scalar_type() == weights.scalar_type(), "Tensor dtype mismatch. Expected ",
                m_weights_.scalar_type(), ", got ", weights.scalar_type());
    m_weights_.copy_(weights);
  }

 private:
  torch::Tensor m_weights_;  ///< Embedding weight matrix, shape [vocab_size, embedding_dim].
  int m_embedding_dim_;      ///< Dimension of the embedding vectors.
};

}  // namespace yllang

#endif  // YLLANG_LAYERS_VOCAB_EMBEDDING_H