/**
 * @file decode_manager.h
 * @brief Defines the DecodeManager class responsible for managing the decode phase.
 */

#ifndef YLLANG_DECODE_MANAGER_H
#define YLLANG_DECODE_MANAGER_H

#include <cstddef>

namespace yllang {

/**
 * @brief Manages the decode (token generation) phase of inference.
 *
 * This class is responsible for orchestrating the decoding loop, including
 * scheduling, memory management, and possibly early stopping. Currently it
 * only declares a size query method.
 */
class DecodeManager {
 public:
  /**
   * @brief Returns the number of tokens that should be retained in the decode buffer.
   *
   * This size may be used to determine how many generated tokens to keep
   * for subsequent iterations or for returning to the user.
   *
   * @return size_t Number of tokens to retain.
   */
  auto SizeToRetained() -> size_t;
};

}  // namespace yllang

#endif  // YLLANG_DECODE_MANAGER_H