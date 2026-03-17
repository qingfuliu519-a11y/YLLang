#ifndef YLLANG_KVCACHE_RADIX_TREE_H_
#define YLLANG_KVCACHE_RADIX_TREE_H_

#include <torch/torch.h>
#include <algorithm>
#include <map>
#include <memory>
#include <optional>
#include <ranges>
#include <sstream>
#include <string>
#include <vector>
namespace yllang {
class RadixNode;

/**
 * @brief 遍历结果句柄
 */
struct RadixHandle {
  torch::Tensor m_values_;                  ///< 匹配的所有Value张量（与keys一一对应）
  std::shared_ptr<RadixNode> m_last_node_;  ///< 最后一个匹配的子节点
  size_t m_matched_length_{0};              ///< 匹配长度
  size_t m_latest_node_matched_len_{0};

  RadixHandle() : m_values_(torch::empty({})), m_last_node_(nullptr) {}

  auto MatchedLength() const -> size_t { return m_matched_length_; }
};

class RadixNode : public std::enable_shared_from_this<RadixNode> {
 public:
  using RadixKeyType = int;
  using RadixValueType = int;
  RadixNode() = default;

  ~RadixNode() = default;

  auto Length() const -> size_t { return m_key_.size(0); }

  auto SetKeyValues(torch::Tensor key, torch::Tensor value) -> void {
    m_key_ = std::move(key);
    m_value_ = std::move(value);
  }

  auto SplitAt(size_t index) -> std::shared_ptr<RadixNode> {
    auto length = this->Length();
    if (length <= index) {
      return nullptr;
    }
    std::shared_ptr<RadixNode> child = std::make_shared<RadixNode>();
    auto keys = m_key_.split_with_sizes({static_cast<int>(index + 1), static_cast<int>(length - index - 1)}, 0);
    auto values = m_value_.split_with_sizes({static_cast<int>(index + 1), static_cast<int>(length - index - 1)}, 0);
    assert(2 == keys.size() && 2 == values.size());

    this->SetKeyValues(keys[1], values[1]);
    child->SetKeyValues(keys[0], values[0]);

    child->m_parent_ = this->m_parent_;
    this->m_parent_ = child;

    // TODO
    child->m_ref_count_ = this->m_ref_count_;

    child->m_children_[keys[1].index({0}).item<RadixKeyType>()] = this->shared_from_this();
    return child;
  }

  auto Retain() -> void { this->m_ref_count_++; }

  auto Release() -> void { this->m_ref_count_--; }

  auto SetParent(const std::shared_ptr<RadixNode> &parent) -> void { m_parent_ = parent; }

  auto Parent() const -> std::shared_ptr<RadixNode> { return m_parent_.lock(); }

  auto Children() -> std::map<RadixKeyType, std::shared_ptr<RadixNode>> & { return m_children_; }

  auto Erase(const std::shared_ptr<RadixNode> &child) -> void {
    m_children_.erase(child->m_key_.index({0}).item<RadixKeyType>());
  }

  auto Value() const -> torch::Tensor { return m_value_; }

  auto RefCount() const -> size_t { return m_ref_count_; }

  auto IsRoot() const -> bool { return m_parent_.expired(); }

  auto IsLeaf() const -> bool { return m_children_.empty(); }

  auto Insert(const std::shared_ptr<RadixNode> &child) -> void {
    assert(child->m_key_.dim() == 1 && child->m_key_.size(0) > 0);
    auto key = child->m_key_.index({0}).item<RadixKeyType>();
    assert(!m_children_.contains(key));
    m_children_[key] = child;
  }

  auto Find(RadixKeyType key) const -> std::shared_ptr<RadixNode> {
    auto it = m_children_.find(key);
    return (it != m_children_.end()) ? it->second : nullptr;
  }

  auto MatchedLen(const torch::Tensor &input) const -> RadixKeyType {
    TORCH_CHECK(input.dim() == 1, "Tensors must be 1D");
    int64_t min_len = std::min(input.size(0), m_key_.size(0));

    auto a_slice = input.slice(0, 0, min_len);
    auto b_slice = m_key_.slice(0, 0, min_len);

    // 直接比较得到布尔掩码
    auto neq = (a_slice != b_slice);

    // 使用 any().item() 快速判断是否有不等
    if (!neq.any().item<bool>()) {
      return static_cast<RadixKeyType>(min_len);
    }

    // 使用 argmax 查找第一个 true 的位置（布尔可直接参与，但需要转换类型）
    // 更好的方式：利用 torch::where 或自定义 CUDA 核；这里用 argmax 并保持 bool 类型
    auto neq_byte = neq.to(torch::kByte);  // 仅需 0/1，比 float 轻量
    auto mismatch_pos = torch::argmax(neq_byte, 0).item<int64_t>();
    return static_cast<RadixKeyType>(mismatch_pos);
  }

 private:
  torch::Tensor m_key_;
  torch::Tensor m_value_;
  std::weak_ptr<RadixNode> m_parent_;
  size_t m_ref_count_{0};
  std::map<RadixKeyType, std::shared_ptr<RadixNode>> m_children_;
};

class RadixTree {
 public:
  RadixTree() : m_root_(std::make_shared<RadixNode>()) {}

  auto EvitableSize() const -> size_t { return m_evitable_size_; }

  auto ProtectedSize() const -> size_t { return m_protected_size_; }

  auto MatchPrefix(const torch::Tensor &prefix) const -> std::shared_ptr<RadixHandle> {
    assert(prefix.dim() == 1);
    std::shared_ptr<RadixHandle> handle = this->Walk(prefix, true);
    return handle;
  }

  auto Insert(const torch::Tensor &key, const torch::Tensor &value) -> size_t {
    assert(key.dim() == 1);
    assert(value.dim() == 1);
    std::shared_ptr<RadixHandle> handle = this->Walk(key, false);
    if (key.size(0) > handle->MatchedLength()) {
      if (handle->m_latest_node_matched_len_ != 0 &&
          handle->m_latest_node_matched_len_ < handle->m_last_node_->Length()) {
        handle->m_last_node_ = handle->m_last_node_->SplitAt(handle->m_latest_node_matched_len_ - 1);
      }
      auto new_node = std::make_shared<RadixNode>();
      new_node->SetKeyValues(key.slice(0, handle->MatchedLength(), key.size(0)),
                             value.slice(0, handle->MatchedLength(), key.size(0)));
      handle->m_last_node_->Insert(new_node);
      new_node->SetParent(handle->m_last_node_);
      this->m_evitable_size_ += new_node->Length();
    }
    return handle->MatchedLength();
  }

  auto Unlock(const std::shared_ptr<RadixHandle> &handle) -> void {
    auto cur_node = handle->m_last_node_;
    while (!cur_node->IsRoot()) {
      cur_node->Release();
      if (cur_node->RefCount() == 0) {
        m_evitable_size_ += cur_node->Length();
        m_protected_size_ -= cur_node->Length();
      }
      cur_node = cur_node->Parent();
    }
  }

  auto Lock(const std::shared_ptr<RadixHandle> &handle) -> void {
    auto cur_node = handle->m_last_node_;
    while (!cur_node->IsRoot()) {
      if (cur_node->RefCount() == 0) {
        m_evitable_size_ -= cur_node->Length();
        m_protected_size_ += cur_node->Length();
      }
      cur_node->Retain();
      cur_node = cur_node->Parent();
    }
  }

  auto Evict(size_t size) -> torch::Tensor {
    if (size <= 0 || size >= m_evitable_size_) {
      return {};
    }

    torch::Tensor evict_res = torch::empty({});
    auto leave_nodes = this->CollectLeaveNodesForEvict();
    while (!leave_nodes.empty() && size > 0) {
      auto node = leave_nodes.back();
      leave_nodes.pop_back();
      evict_res = torch::cat({node->Value(), evict_res}, 0);
      m_evitable_size_ -= node->Length();
      size -= node->Length();

      auto parent = node->Parent();
      parent->Erase(node);
      if (parent->IsLeaf() && parent->RefCount() == 0) {
        leave_nodes.push_back(parent);
      }
    }
    return evict_res;
  }

 protected:
  auto Walk(const torch::Tensor &prefix, bool valuesNeeded) const -> std::shared_ptr<RadixHandle> {
    assert(prefix.dim() == 1);
    if (prefix.size(0) == 0) {
      return nullptr;
    }
    std::shared_ptr<RadixHandle> handle = std::make_shared<RadixHandle>();
    auto cur_node = m_root_;
    auto remained_len = prefix.size(0);
    auto cur_key = prefix.index({static_cast<int64_t>(handle->m_matched_length_)}).item<RadixNode::RadixKeyType>();
    auto next_node = cur_node->Find(cur_key);
    while (next_node != nullptr) {
      handle->m_latest_node_matched_len_ =
          next_node->MatchedLen(prefix.slice(0, handle->m_matched_length_, prefix.size(0)));
      if (valuesNeeded) {
        handle->m_values_ =
            torch::cat({handle->m_values_, next_node->Value().slice(0, 0, handle->m_latest_node_matched_len_)}, 0);
      }
      handle->m_matched_length_ += handle->m_latest_node_matched_len_;
      remained_len -= handle->m_latest_node_matched_len_;
      cur_node = next_node;
      if (remained_len == 0 || handle->m_latest_node_matched_len_ != next_node->Length()) {
        break;
      }
      cur_key = prefix.index({static_cast<int64_t>(handle->m_matched_length_)}).item<RadixNode::RadixKeyType>();
      next_node = cur_node->Find(cur_key);
    }
    handle->m_last_node_ = cur_node;
    return handle;
  }

  auto CollectLeaveNodesForEvict() const -> std::vector<std::shared_ptr<RadixNode>> {
    std::vector<std::shared_ptr<RadixNode>> nodes;
    std::vector<std::shared_ptr<RadixNode>> leave_nodes;
    nodes.push_back(m_root_);
    while (!nodes.empty()) {
      auto cur_node = nodes.back();
      nodes.pop_back();
      for (auto &children = cur_node->Children(); auto &v : children | std::views::values) {
        if (v->IsLeaf() && !v->IsRoot()) {
          if (v->RefCount() == 0) {
            leave_nodes.push_back(v);
          }
        } else {
          nodes.push_back(v);
        }
      }
    }
    return leave_nodes;
  }

 private:
  std::shared_ptr<RadixNode> m_root_;
  size_t m_evitable_size_{0};
  size_t m_protected_size_{0};
};
using RadixIndicesManager = RadixTree;
}  // namespace yllang

#endif
