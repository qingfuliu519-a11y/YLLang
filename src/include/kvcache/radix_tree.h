/**
 * @file radix_tree.h
 * @brief Defines a radix tree (trie) for managing KV cache indices with prefix matching and eviction.
 */

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
 * @brief Handle returned by prefix matching operations.
 *
 * Contains the concatenated values of all matched nodes, the last node reached,
 * and the total matched length.
 */
struct RadixHandle {
  torch::Tensor m_values_;  ///< Concatenated values of all matched nodes (aligned with the matched keys).
  std::shared_ptr<RadixNode> m_last_node_;  ///< Last node visited during the walk.
  size_t m_matched_length_{0};              ///< Total number of matched tokens (cumulative length).
  size_t m_latest_node_matched_len_{0};     ///< Length matched within the last node.

  RadixHandle() : m_values_(torch::empty({})), m_last_node_(nullptr) {}

  /**
   * @brief Returns the total matched length.
   *
   * @return size_t Matched length.
   */
  auto MatchedLength() const -> size_t { return m_matched_length_; }
};

/**
 * @brief Node of the radix tree.
 *
 * Each node stores a contiguous segment of a key (token IDs) and its corresponding values
 * (e.g., page indices). Nodes can be split when a partial match occurs, and maintain
 * reference counts for eviction policy.
 */
class RadixNode : public std::enable_shared_from_this<RadixNode> {
 public:
  using RadixKeyType = int;    ///< Type of a single token ID.
  using RadixValueType = int;  ///< Type of a stored value (e.g., page index).

  RadixNode() = default;
  ~RadixNode() = default;

  /**
   * @brief Returns the length of the key stored in this node.
   *
   * @return size_t Number of tokens in the key segment.
   */
  auto Length() const -> size_t { return m_key_.size(0); }

  /**
   * @brief Sets the key and value tensors for this node.
   *
   * Both tensors must be 1D and have the same length.
   *
   * @param key   1D tensor of token IDs.
   * @param value 1D tensor of corresponding values.
   */
  auto SetKeyValues(torch::Tensor key, torch::Tensor value) -> void {
    m_key_ = std::move(key);
    m_value_ = std::move(value);
  }

  /**
   * @brief Splits the node at the given index, creating a new parent node.
   *
   * The current node becomes a child of the newly created node. This is used when a
   * prefix match ends in the middle of a node's key.
   *
   * @param index Split position (0‑based) within the key. Must be less than Length().
   * @return std::shared_ptr<RadixNode> The new parent node, or nullptr if index is invalid.
   */
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

    // Transfer reference count to the new parent (simplified; may need adjustment)
    child->m_ref_count_ = this->m_ref_count_;

    child->m_children_[keys[1].index({0}).item<RadixKeyType>()] = this->shared_from_this();
    return child;
  }

  /// Increments the reference count.
  auto Retain() -> void { this->m_ref_count_++; }

  /// Decrements the reference count.
  auto Release() -> void { this->m_ref_count_--; }

  /// Sets the parent node.
  auto SetParent(const std::shared_ptr<RadixNode> &parent) -> void { m_parent_ = parent; }

  /// Returns the parent node (may be null if this is the root).
  auto Parent() const -> std::shared_ptr<RadixNode> { return m_parent_.lock(); }

  /// Returns a reference to the children map (keyed by the first token of each child's key).
  auto Children() -> std::map<RadixKeyType, std::shared_ptr<RadixNode>> & { return m_children_; }

  /// Removes a child node from the children map.
  auto Erase(const std::shared_ptr<RadixNode> &child) -> void {
    m_children_.erase(child->m_key_.index({0}).item<RadixKeyType>());
  }

  /// Returns the value tensor of this node.
  auto Value() const -> torch::Tensor { return m_value_; }

  /// Returns the current reference count.
  auto RefCount() const -> size_t { return m_ref_count_; }

  /// Checks whether this node is the root (has no parent).
  auto IsRoot() const -> bool { return m_parent_.expired(); }

  /// Checks whether this node has no children.
  auto IsLeaf() const -> bool { return m_children_.empty(); }

  /**
   * @brief Inserts a child node.
   *
   * The child's first token must be unique among existing children.
   *
   * @param child The child node to insert.
   */
  auto Insert(const std::shared_ptr<RadixNode> &child) -> void {
    assert(child->m_key_.dim() == 1 && child->m_key_.size(0) > 0);
    auto key = child->m_key_.index({0}).item<RadixKeyType>();
    assert(!m_children_.contains(key));
    m_children_[key] = child;
  }

  /**
   * @brief Finds a child node by its first token.
   *
   * @param key The first token of the child's key.
   * @return std::shared_ptr<RadixNode> The child node, or nullptr if not found.
   */
  auto Find(RadixKeyType key) const -> std::shared_ptr<RadixNode> {
    auto it = m_children_.find(key);
    return (it != m_children_.end()) ? it->second : nullptr;
  }

  /**
   * @brief Computes the length of the common prefix between the node's key and the input tensor.
   *
   * @param input 1D tensor of token IDs.
   * @return RadixKeyType Number of matching tokens (from the start).
   */
  auto MatchedLen(const torch::Tensor &input) const -> RadixKeyType {
    TORCH_CHECK(input.dim() == 1, "Tensors must be 1D");
    int64_t min_len = std::min(input.size(0), m_key_.size(0));

    auto a_slice = input.slice(0, 0, min_len);
    auto b_slice = m_key_.slice(0, 0, min_len);

    // Compare elementwise to get a boolean mask.
    auto neq = (a_slice != b_slice);

    // If no inequality found, the whole slice matches.
    if (!neq.any().item<bool>()) {
      return static_cast<RadixKeyType>(min_len);
    }

    // Convert to byte for argmax (to find first true position).
    auto neq_byte = neq.to(torch::kByte);
    auto mismatch_pos = torch::argmax(neq_byte, 0).item<int64_t>();
    return static_cast<RadixKeyType>(mismatch_pos);
  }

 private:
  torch::Tensor m_key_;                                            ///< Key segment (token IDs).
  torch::Tensor m_value_;                                          ///< Value segment (page indices).
  std::weak_ptr<RadixNode> m_parent_;                              ///< Parent node (weak to avoid cycles).
  size_t m_ref_count_{0};                                          ///< Reference count for eviction.
  std::map<RadixKeyType, std::shared_ptr<RadixNode>> m_children_;  ///< Child nodes indexed by first token.
};

/**
 * @brief Radix tree (trie) for managing KV cache indices.
 *
 * Supports prefix matching, insertion, locking/unlocking of handles, and eviction
 * of nodes with zero reference count.
 */
class RadixTree {
 public:
  RadixTree() : m_root_(std::make_shared<RadixNode>()) {}

  /// Returns the total number of tokens stored in evictable nodes.
  auto EvitableSize() const -> size_t { return m_evitable_size_; }

  /// Returns the total number of tokens stored in protected (locked) nodes.
  auto ProtectedSize() const -> size_t { return m_protected_size_; }

  /**
   * @brief Matches the longest prefix of the input tensor in the tree.
   *
   * @param prefix 1D tensor of token IDs.
   * @return std::shared_ptr<RadixHandle> Handle containing matched information.
   */
  auto MatchPrefix(const torch::Tensor &prefix) const -> std::shared_ptr<RadixHandle> {
    assert(prefix.dim() == 1);
    std::shared_ptr<RadixHandle> handle = this->Walk(prefix, true);
    return handle;
  }

  /**
   * @brief Inserts a key–value pair into the tree.
   *
   * If the key already partially exists, the tree is extended accordingly.
   * The new node(s) are added as evictable.
   *
   * @param key   1D tensor of token IDs.
   * @param value 1D tensor of corresponding values (same length as key).
   * @return size_t The length of the already existing prefix.
   */
  auto Insert(const torch::Tensor &key, const torch::Tensor &value) -> size_t {
    assert(key.dim() == 1);
    assert(value.dim() == 1);
    std::shared_ptr<RadixHandle> handle = this->Walk(key, false);
    if (key.size(0) > handle->MatchedLength()) {
      // If the match ended in the middle of a node, split that node.
      if (handle->m_latest_node_matched_len_ != 0 &&
          handle->m_latest_node_matched_len_ < handle->m_last_node_->Length()) {
        handle->m_last_node_ = handle->m_last_node_->SplitAt(handle->m_latest_node_matched_len_ - 1);
      }
      // Create a new node for the unmatched suffix.
      auto new_node = std::make_shared<RadixNode>();
      new_node->SetKeyValues(key.slice(0, handle->MatchedLength(), key.size(0)),
                             value.slice(0, handle->MatchedLength(), key.size(0)));
      handle->m_last_node_->Insert(new_node);
      new_node->SetParent(handle->m_last_node_);
      this->m_evitable_size_ += new_node->Length();
    }
    return handle->MatchedLength();
  }

  /**
   * @brief Unlocks a previously locked handle, making its nodes evictable if reference count reaches zero.
   *
   * @param handle The handle to unlock.
   */
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

  /**
   * @brief Locks a handle, preventing its nodes from being evicted.
   *
   * @param handle The handle to lock.
   */
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

  /**
   * @brief Evicts nodes to free up approximately `size` tokens.
   *
   * Collects leaf nodes with zero reference count and removes them until the requested
   * size is met. The values of evicted nodes are concatenated and returned.
   *
   * @param size Number of tokens to evict.
   * @return torch::Tensor Concatenated values of the evicted nodes.
   */
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
  /**
   * @brief Walks the tree following the prefix, optionally collecting values.
   *
   * @param prefix      1D tensor of token IDs.
   * @param valuesNeeded If true, values of matched nodes are concatenated into the handle.
   * @return std::shared_ptr<RadixHandle> Handle containing matched information.
   */
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

  /**
   * @brief Collects all leaf nodes that are evictable (reference count zero).
   *
   * @return std::vector<std::shared_ptr<RadixNode>> List of evictable leaf nodes.
   */
  auto CollectLeaveNodesForEvict() const -> std::vector<std::shared_ptr<RadixNode>> {
    std::vector<std::shared_ptr<RadixNode>> nodes;
    std::vector<std::shared_ptr<RadixNode>> leave_nodes;
    nodes.push_back(m_root_);
    while (!nodes.empty()) {
      auto cur_node = nodes.back();
      nodes.pop_back();
      auto &children = cur_node->Children();
      for (auto &[_, v] : children) {
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
  std::shared_ptr<RadixNode> m_root_;  ///< Root node (empty key).
  size_t m_evitable_size_{0};          ///< Total number of tokens in evictable nodes.
  size_t m_protected_size_{0};         ///< Total number of tokens in locked nodes.
};

/// Alias for RadixTree used as indices manager.
using RadixIndicesManager = RadixTree;

}  // namespace yllang

#endif  // YLLANG_KVCACHE_RADIX_TREE_H_