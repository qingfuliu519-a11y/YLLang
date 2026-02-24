#ifndef YLLANG_UTIL_PANIC_H_
#define YLLANG_UTIL_PANIC_H_
#include <source_location>
#include <sstream>
#include <utility>

namespace yllang {
class PanicError final : public std::runtime_error {
 public:
  PanicError(std::string msg) : std::runtime_error(msg), m_message(std::move(msg)) {}

  auto detail() const -> std::string_view {
    const auto sv = std::string_view(m_message);
    const auto pos = sv.find(": ");
    return pos == std::string_view::npos ? sv : sv.substr(pos + 2);
  }

 private:
  std::string m_message;
};

template <typename... Args>
[[noreturn]] inline auto Panic(std::source_location location, Args &&...args) -> void {
  std::ostringstream oss;
  oss << "Runtime check failed At:" << location.file_name() << ":" << location.line();
  oss << " (" << location.function_name() << ")";
  if constexpr (sizeof...(args) > 0) {
    oss << " : ";
    (oss << ... << std::forward<Args>(std::move(args)));
  }
  throw PanicError(std::move(oss).str());
}

}  // namespace yllang

#endif  // YLLANG_UTIL_PANIC_H_
