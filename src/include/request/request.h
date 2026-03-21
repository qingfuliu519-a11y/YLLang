/**
 * @file request.h
 * @brief Defines classes representing user messages, active requests, and pending requests.
 */

#ifndef YLLANG_REQUEST_H
#define YLLANG_REQUEST_H

#include <torch/torch.h>
#include <string>
#include "kvcache/radix_tree.h"

namespace yllang {

/**
 * @brief Represents a user message containing the raw input text and a user identifier.
 *
 * This class is used to encapsulate the user's original message before tokenization
 * and processing.
 */
class UserMsg {
 public:
  UserMsg() = default;

  ~UserMsg() = default;

  /**
   * @brief Constructs a UserMsg with the given input message and user ID.
   *
   * @param input_msg The raw input text from the user.
   * @param user_id   String identifying the user.
   */
  UserMsg(std::string user_role, std::string input_msg)
      : m_input_msg_(std::move(input_msg)), m_user_role_(std::move(user_role)) {}

  UserMsg(const UserMsg &) = default;
  auto operator=(const UserMsg &) -> UserMsg & = default;

  /// Returns a const reference to the input message.
  auto InputMsg() const -> const std::string & { return m_input_msg_; }

  /// Returns a mutable reference to the input message.
  auto InputMsg() -> std::string & { return m_input_msg_; }

  /// Sets the input message.
  auto SetInputIds(std::string msg) -> void { m_input_msg_ = std::move(msg); }

  /// Returns a const reference to the user ID string.
  auto UserRole() const -> std::string { return m_user_role_; }

  /// Returns a mutable reference to the user ID string.
  auto UserRole() -> std::string & { return m_user_role_; }

  /// Sets the user ID string.
  auto SetUserId(std::string user_role) -> void { m_user_role_ = std::move(user_role); }

 private:
  std::string m_input_msg_;  ///< Raw user input message.
  std::string m_user_role_;  ///< User identifier.
};

/**
 * @brief Represents an active request that is currently being processed.
 *
 * Contains the input token IDs, caching information, and a handle to the
 * cached radix tree node. It tracks the length of the input that was already
 * cached and the length that will be processed on the device.
 */
class Request {
 public:
  /**
   * @brief Constructs a Request.
   *
   * @param input_ids      Full input token IDs (including cached and new parts).
   * @param user_id        User identifier.
   * @param cached_len     Number of tokens already cached.
   * @param table_index    Index in the cache table (slot) assigned to this request.
   * @param output_len     Expected length of the generated output.
   * @param cached_handle  Radix handle for the cached prefix.
   */
  Request(torch::Tensor input_ids, std::string user_id, const size_t &cached_len, const size_t &table_index,
          const size_t &output_len, std::shared_ptr<RadixHandle> cached_handle)
      : m_input_ids_(std::move(input_ids)),
        m_user_id_(std::move(user_id)),
        m_device_len_(input_ids.size(0)),
        m_cached_len_(cached_len),
        m_table_index_(table_index),
        m_output_len_(output_len),
        m_cached_handle_(std::move(cached_handle)) {}

  /// Returns the length of new tokens to be processed (device length - cached length).
  auto ExtendLen() const -> size_t { return m_device_len_ - m_cached_len_; }

  /// Returns the cache table slot index.
  auto TableIndex() const -> size_t { return m_table_index_; }

  /// Returns the number of tokens already cached.
  auto CachedLen() const -> size_t { return m_cached_len_; }

  /// Returns the total length of input tokens on the device.
  auto DeviceLen() const -> size_t { return m_device_len_; }

 private:
  torch::Tensor m_input_ids_;                     ///< Full input token IDs.
  std::string m_user_id_;                         ///< User identifier.
  size_t m_device_len_;                           ///< Total length of input on device.
  size_t m_cached_len_;                           ///< Length of already cached prefix.
  size_t m_table_index_;                          ///< Cache table slot index.
  size_t m_output_len_;                           ///< Expected output length.
  std::shared_ptr<RadixHandle> m_cached_handle_;  ///< Handle to cached radix node.
};

/**
 * @brief Represents a pending request waiting to be scheduled.
 *
 * Contains the full input and expected output length. It is used by the batch
 * manager to build batches.
 */
class PaddingRequest {
 public:
  /**
   * @brief Constructs a PaddingRequest.
   *
   * @param input_ids   Full input token IDs.
   * @param user_id     User identifier.
   * @param output_len  Expected output length.
   */
  PaddingRequest(torch::Tensor input_ids, std::string user_id, size_t output_len)
      : m_input_ids_(std::move(input_ids)), m_user_id_(std::move(user_id)), m_output_len_(output_len) {}

  /// Returns the length of the input.
  auto InputLen() const -> size_t { return m_input_ids_.size(0); }

  /// Returns the user ID.
  auto UserId() const -> std::string { return m_user_id_; }

  /// Returns a const reference to the input IDs tensor.
  auto InputIds() const -> const torch::Tensor & { return m_input_ids_; }

  /// Returns the expected output length.
  auto OutputLen() const -> size_t { return m_output_len_; }

 private:
  torch::Tensor m_input_ids_;  ///< Full input token IDs.
  std::string m_user_id_;      ///< User identifier.
  size_t m_output_len_;        ///< Expected output length.
};

}  // namespace yllang

#endif  // YLLANG_REQUEST_H