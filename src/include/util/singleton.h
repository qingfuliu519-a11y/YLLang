#ifndef YLLANG_UTIL_Singleton_H
#define YLLANG_UTIL_Singleton_H

namespace yllang {

namespace util {

template <typename Derived>
class Singleton {
 public:
  Singleton(const Singleton &) = delete;
  Singleton &operator=(const Singleton &) = delete;
  Singleton(Singleton &&) = delete;
  Singleton &operator=(Singleton &&) = delete;

  static Derived &Instance() { return m_instance_; }

 protected:
  Singleton() = default;
  virtual ~Singleton() = default;
  inline static Derived m_instance_;
};

}  // namespace util

}  // namespace yllang

#endif  // YLLANG_UTIL_Singleton_H
