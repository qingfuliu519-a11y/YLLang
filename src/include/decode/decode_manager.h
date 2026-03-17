#ifndef YLLANG_DECODE_MANAGER_H
#define YLLANG_DECODE_MANAGER_H
#include <cstddef>

namespace yllang {
class DecodeManager {
 public:
  auto SizeToRetained() -> size_t;
};
}  // namespace yllang

#endif  // YLLANG_DECODE_MANAGER_H
