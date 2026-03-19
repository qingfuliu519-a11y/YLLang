#ifndef YLLANG_UTIL_UTIL_H_
#define YLLANG_UTIL_UTIL_H_

#include <cstdint>
#include <fstream>
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

namespace util {

auto LoadBytesFromFile(const std::string &path) -> std::string {
  std::ifstream fs(path, std::ios::in | std::ios::binary);
  if (fs.fail()) {
    std::cerr << "Cannot open " << path << std::endl;
    exit(1);
  }
  std::string data;
  fs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(fs.tellg());
  fs.seekg(0, std::ios::beg);
  data.resize(size);
  fs.read(data.data(), size);
  return data;
}

}  // namespace util

}  // namespace yllang
#endif  // YLLANG_UTIL_UTIL_H_
