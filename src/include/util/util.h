#ifndef YLLANG_UTIL_UTIL_H_
#define YLLANG_UTIL_UTIL_H_

#include <cstdint>

namespace yllang {
template <typename T>
class BaseRef {
 public:
  BaseRef(const BaseRef &) = delete;
  auto operator=(const BaseRef &) -> BaseRef & = delete;

  BaseRef() { m_ref_ = &m_placeholder_; }
  BaseRef(T &t) : m_ref_(&t), m_placeholder_() {}

  auto operator->() const -> T * { return m_ref_; }
  auto operator*() const -> T & { return *m_ref_; }

  void Rebind(T &t) { m_ref_ = &t; }

 private:
  T *m_ref_ = nullptr;
  T m_placeholder_{};
};

}  // namespace yllang
#endif  // YLLANG_UTIL_UTIL_H_
