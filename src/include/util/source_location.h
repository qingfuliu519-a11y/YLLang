#ifndef YLLANG_UTIL_SOURCE_LOCATION_H
#define YLLANG_UTIL_SOURCE_LOCATION_H
#include <algorithm>

namespace yllang {
class SourceLocation {
 public:
  SourceLocation() = default;

  SourceLocation(std::string filename, const int line) : m_file_name_(std::move(filename)), m_line_(line) {}

  ~SourceLocation() = default;

  auto FileName() const -> const std::string & { return m_file_name_; }

  auto Line() const -> int { return m_line_; }

  static auto Current(const std::string &filename = __FILE__, const int line = __LINE__) -> SourceLocation {
    return {filename, line};
  }

 private:
  std::string m_file_name_;
  int m_line_{-1};
};
}  // namespace yllang

#endif  // YLLANG_UTIL_SOURCE_LOCATION_H
