#ifndef YLLANG_REQUEST_H
#define YLLANG_REQUEST_H
#include <torch/torch.h>
#include <string>
#include "kvcache/radix_tree.h"

namespace yllang {
class UserMsg {
 public:
  UserMsg() = default;

  ~UserMsg() = default;

  UserMsg(torch::Tensor input_ids, std::string user_id)
      : m_input_ids_(std::move(input_ids)), m_user_id_(std::move(user_id)) {}

  UserMsg(const UserMsg &) = delete;

  auto operator=(const UserMsg &) = delete;

  auto InputIds() const -> torch::Tensor { return m_input_ids_; }

  auto InputIds() -> torch::Tensor & { return m_input_ids_; }

  auto SetInputIds(torch::Tensor ids) -> void { m_input_ids_ = std::move(ids); }

  auto UserId() const -> std::string { return m_user_id_; }

  auto UserId() -> std::string & { return m_user_id_; }

  auto SetUserId(std::string user_id) -> void { m_user_id_ = std::move(user_id); }

 private:
  torch::Tensor m_input_ids_;
  std::string m_user_id_;
};

class Request {
 public:
  Request(torch::Tensor input_ids, std::string user_id, const size_t &cached_len, const size_t &table_index,
          const size_t &output_len, std::shared_ptr<RadixHandle> cached_handle)
      : m_input_ids_(std::move(input_ids)),
        m_user_id_(std::move(user_id)),
        m_device_len_(input_ids.size(0)),
        m_cached_len_(cached_len),
        m_table_index_(table_index),
        m_output_len_(output_len),
        m_cached_handle_(std::move(cached_handle)) {}

  auto ExtendLen() const -> size_t { return m_device_len_ - m_cached_len_; }

  auto TableIndex() const -> size_t { return m_table_index_; }

  auto CachedLen() const -> size_t { return m_cached_len_; }

  auto DeviceLen() const -> size_t { return m_device_len_; }

 private:
  torch::Tensor m_input_ids_;
  std::string m_user_id_;
  size_t m_device_len_;
  size_t m_cached_len_;
  size_t m_table_index_;
  size_t m_output_len_;
  std::shared_ptr<RadixHandle> m_cached_handle_;
};

class PaddingRequest {
 public:
  PaddingRequest(torch::Tensor input_ids, std::string user_id, size_t output_len)
      : m_input_ids_(std::move(input_ids)), m_user_id_(std::move(user_id)), m_output_len_(output_len) {}

  auto InputLen() const -> size_t { return m_input_ids_.size(0); }

  auto UserId() const -> std::string { return m_user_id_; }

  auto InputIds() const -> const torch::Tensor & { return m_input_ids_; }

  auto OutputLen() const -> size_t { return m_output_len_; }

 private:
  torch::Tensor m_input_ids_;
  std::string m_user_id_;
  size_t m_output_len_;
};
}  // namespace yllang

#endif  // YLLANG_REQUEST_H
