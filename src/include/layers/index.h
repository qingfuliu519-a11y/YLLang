/**
 * @file index.h
 * @brief Defines the embedding lookup kernel and its host launcher.
 *
 * This file provides a high‑performance CUDA kernel for gathering rows from a
 * weight matrix (embedding table) according to given indices and storing them
 * into a destination tensor. It is used in the embedding layer of large language
 * models to map token IDs to dense vector representations.
 */

#ifndef YLLANG_LAYERS_INDEX_H
#define YLLANG_LAYERS_INDEX_H

#include <torch/torch.h>
#include <concepts>
#include <cstdint>

namespace yllang {

/**
 * @brief Gathers rows from an embedding weight tensor into a destination tensor.
 *
 * For each index in `indices`, the corresponding row from `weights` is copied
 * into the output `destination`. The operation is parallelized over warps and
 * uses vectorized memory transfers for optimal throughput.
 *
 * @tparam kElementSize      Total number of bytes to copy per row (must equal
 *                           weights.size(1) * dtype_size).
 * @tparam kThreadsPreBlock  Number of threads per CUDA block (default 128).
 * @tparam kUsePDL           Whether to use programmatic stream serialization.
 * @param indices           1D tensor of token IDs (int32 or int64).
 * @param weights           2D tensor of shape [vocab_size, hidden_dim] – the
 *                          embedding table.
 * @param destination       2D tensor of shape [num_indices, hidden_dim] – the
 *                          output buffer.
 */
template <size_t kElementSize, size_t kThreadsPreBlock = 128, bool kUsePDL = false>
auto Index(const torch::Tensor &indices, const torch::Tensor &weights, const torch::Tensor &destination) -> void;

}  // namespace yllang

#endif  // YLLANG_LAYERS_INDEX_H