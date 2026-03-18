#ifndef YLLANG_UTIL_PANIC_H_
#define YLLANG_UTIL_PANIC_H_
#include <sstream>
#include <utility>
#include "util/source_location.h"

namespace yllang {
class PanicError final : public std::runtime_error {
 public:
  PanicError(std::string msg) : std::runtime_error(msg), m_message_(std::move(msg)) {}

  auto Detail() const -> std::string_view {
    const auto sv = std::string_view(m_message_);
    const auto pos = sv.find(": ");
    return pos == std::string_view::npos ? sv : sv.substr(pos + 2);
  }

 private:
  std::string m_message_;
};

template <typename... Args>
[[noreturn]] inline auto Panic(const yllang::SourceLocation &location, Args &&...args) -> void {
  std::ostringstream oss;
  oss << "Runtime check failed At:" << location.FileName() << ":" << location.Line();
  // oss << " (" << location.function_name() << ")";
  if constexpr (sizeof...(args) > 0) {
    oss << " : ";
    // (oss << ... << std::forward<Args>(std::move(args)));
  }
  throw PanicError(std::move(oss).str());
}

template <typename... Args>
class RuntimeCheck {
 public:
  template <typename T>
  explicit RuntimeCheck(T &&condition, Args &&...args,
                        yllang::SourceLocation location = yllang::SourceLocation::Current()) {
    if (!condition) {
      [[unlikely]];
      Panic(location, std::forward<Args>(args)...);
    }
  }
};

template <typename T, typename... Args>
explicit RuntimeCheck(T &&, Args &&...) -> RuntimeCheck<Args...>;
}  // namespace yllang

#endif  // YLLANG_UTIL_PANIC_H_
