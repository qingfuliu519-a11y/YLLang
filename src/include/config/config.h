/**
 * @file config.h
 * @brief Auto‑generated configuration constants from YAML settings. DO NOT EDIT.
 */

#ifndef YLLANG_CONFIG_H_
#define YLLANG_CONFIG_H_

#include <string>

namespace yllang {

/**
 * @brief Size (in bytes) of a single KV cache element.
 *
 * Computed as: head_dim * number_of_kv_heads * dtype_size.
 * Currently derived from YAML as 8 * 64 * 4.
 */
constexpr int kElementSize = 8 * 64 * 4;

/**
 * @brief Number of threads per block for CUDA kernels.
 *
 * This value is used in kernel launches (e.g., StoreKVCacheKernel).
 */
constexpr int kThreadsPreBlock = 128;

/**
 * @brief Number of threads per warp on NVIDIA GPUs.
 *
 * Always 32 for current architectures.
 */
constexpr int kThreadsPreWrap = 32;

const std::string kCachePath = "~/.yllang";

const std::string kModelPath = kCachePath + "/model";

}  // namespace yllang

#endif  // YLLANG_CONFIG_H_