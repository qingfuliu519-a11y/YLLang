
#ifndef YLLANG_UTIL_TENSOR_H
#define YLLANG_UTIL_TENSOR_H

#include <optional>
#include <ranges>
#include <span>
#include "c10/core/Device.h"
#include "c10/util/typeid.h"
#include "tensor.h"
#include "util/panic.h"
#include "util/util.h"
namespace yllang {

constexpr auto kAnySize = static_cast<int64_t>(-1);
constexpr auto K_ANY_SIZE = static_cast<int64_t>(0);

constexpr auto K_NULL_DEVICE = static_cast<c10::DeviceType>(c10::COMPILE_TIME_MAX_DEVICE_TYPES);
constexpr auto K_ANY_DEVICE_ID = static_cast<c10::DeviceIndex>(-1);

class SymbolicSize {
 public:
  SymbolicSize() = default;

  auto HasValue() const -> bool { return m_size_ != yllang::K_ANY_SIZE; }

  auto SetValue(const int64_t &size) -> void {
    RuntimeCheck(!HasValue(), "SymbolicSize::SetValue check failed");
    m_size_ = size;
  }

  auto GetValue() -> std::optional<int64_t> { return HasValue() ? std::optional{m_size_} : std::nullopt; }

  auto UnWrap() const -> int64_t {
    RuntimeCheck(HasValue(), "SymbolicSize::UnWrap check failed");
    return m_size_;
  }

  auto Verify(const int64_t &size) -> void {
    if (HasValue()) {
      RuntimeCheck(size == m_size_, "SymbolicSize::Verify check failed");
    } else {
      SetValue(size);
    }
  }

 private:
  int64_t m_size_{yllang::K_ANY_SIZE};
};

class SymbolicDevice {
 public:
  SymbolicDevice() : m_device_(yllang::K_NULL_DEVICE, yllang::K_ANY_DEVICE_ID) {}

  auto HasValue() const -> bool { return yllang::K_NULL_DEVICE != m_device_.type(); }

  auto GetValue() const -> std::optional<c10::Device> { return HasValue() ? std::optional{m_device_} : std::nullopt; }

  auto SetValue(const c10::Device &device) -> void {
    RuntimeCheck(!HasValue(), "SymbolicDType::SetValue check failed");
    m_device_ = device;
  }

  auto UnWrap() const -> c10::Device {
    RuntimeCheck(HasValue(), "SymbolicDType::UnWrap check failed");
    return m_device_;
  }

  auto Verify(const c10::Device &device) -> void {
    if (HasValue()) {
      RuntimeCheck(m_device_ == device, "SymbolicSize::Verify check failed");
    } else {
      SetValue(device);
    }
  }

 private:
  c10::Device m_device_;
};

class SymbolicDType {
 public:
  SymbolicDType() = default;

  auto HasValue() const -> bool { return caffe2::TypeIdentifier::uninitialized() != m_dtype_.id(); }

  auto GetValue() const -> std::optional<caffe2::TypeMeta> {
    return HasValue() ? std::optional{m_dtype_} : std::nullopt;
  }

  auto SetValue(caffe2::TypeMeta dtype) -> void {
    RuntimeCheck(!HasValue(), "SymbolicDType::SetValue check failed");
    m_dtype_ = dtype;
  }

  auto UnWrap() const -> caffe2::TypeMeta {
    RuntimeCheck(HasValue(), "SymbolicDType::UnWrap check failed");
    return m_dtype_;
  }

  auto Verify(caffe2::TypeMeta dtype) -> void {
    if (HasValue()) {
      RuntimeCheck(dtype == m_dtype_, "SymbolicSize::Verify check failed");
    } else {
      SetValue(dtype);
    }
  }

 private:
  caffe2::TypeMeta m_dtype_;
};

class SymbolicSizeRef : public BaseRef<yllang::SymbolicSize> {
 public:
  SymbolicSizeRef() = default;

  SymbolicSizeRef(int64_t size) {
    if (size != kAnySize) {
      (*this)->SetValue(size);
    } else {
    }
  }

  SymbolicSizeRef(yllang::SymbolicSize &size) : BaseRef<yllang::SymbolicSize>(size) {}
};

class SymbolicDTypeRef : public BaseRef<yllang::SymbolicDType> {
 public:
};

class SymbolicDeviceRef : public BaseRef<yllang::SymbolicDevice> {
 public:
};

class TensorMatcher {
 public:
  TensorMatcher(const TensorMatcher &) = delete;
  auto operator=(const TensorMatcher &) -> TensorMatcher & = delete;

  explicit TensorMatcher(std::initializer_list<SymbolicSizeRef> shape) : m_shape_(shape) {}

  auto WithStride(std::initializer_list<SymbolicSizeRef> stride) && -> TensorMatcher && {
    RuntimeCheck(m_strides_.empty(), "");
    RuntimeCheck(stride.size() == m_shape_.size());
    m_strides_ = stride;
    return std::move(*this);
  }

  auto WithDType(SymbolicDType &dtype) && -> TensorMatcher && {
    InitType();
    m_dtype_.Rebind(dtype);
    return std::move(*this);
  }

  auto WithDevice(SymbolicDevice &device) && -> TensorMatcher && {
    InitDevice();
    m_device_.Rebind(device);
    return std::move(*this);
  }

  auto Verify(const torch::Tensor &view) && -> TensorMatcher && {
    const auto dim = static_cast<std::size_t>(view.dim());
    RuntimeCheck(dim == m_shape_.size());
    for (const auto i : std::views::iota(std::size_t{0}, dim)) {
      m_shape_[i]->Verify(view.size(i));
    }
    if (HasStride()) {
      for (const auto i : std::views::iota(std::size_t{0}, dim)) {
        if (view.size(i) != 1 || !m_strides_[i]->HasValue()) {
          m_strides_[i]->Verify(view.stride(i));
        }
      }
    }

    m_dtype_->Verify(view.dtype());
    m_device_->Verify(view.device());
    return std::move(*this);
  }

 private:
  auto InitType() const -> void { RuntimeCheck(!m_dtype_->HasValue(), ""); }
  auto InitDevice() const -> void { RuntimeCheck(!m_device_->HasValue(), ""); }
  auto HasStride() const -> bool { return !m_strides_.empty(); }

 private:
  std::span<const yllang::SymbolicSizeRef> m_shape_;
  std::span<const yllang::SymbolicSizeRef> m_strides_;
  yllang::SymbolicDTypeRef m_dtype_{};
  yllang::SymbolicDeviceRef m_device_{};
};

}  // namespace yllang

#endif  // YLLANG_UTIL_TENSOR_H
