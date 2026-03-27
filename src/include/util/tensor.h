/**
 * @file util/tensor.h
 * @brief Defines symbolic size, device, dtype classes and a tensor matcher for shape/stride/device/type validation.
 *
 * This file provides utilities for symbolic representation of tensor properties (size, device, dtype)
 * and a TensorMatcher class that can verify actual tensors against these symbolic expectations.
 */

#ifndef YLLANG_UTIL_TENSOR_H
#define YLLANG_UTIL_TENSOR_H

#include <optional>
#include <ranges>
#include <span>
#include "c10/core/Device.h"
#include "c10/util/typeid.h"
#include "safetensors.hh"
#include "tensor.h"
#include "util/panic.h"
#include "util/util.h"
namespace yllang {

/**
 * @brief Sentinel value representing an unspecified size (symbolic).
 */
constexpr auto kAnySize = static_cast<int64_t>(-1);

/**
 * @brief Sentinel device type representing no device (uninitialized).
 */
constexpr auto kNullDevice = static_cast<c10::DeviceType>(c10::COMPILE_TIME_MAX_DEVICE_TYPES);

/**
 * @brief Sentinel device index representing any/unset device index.
 */
constexpr auto kAnyDeviceId = static_cast<c10::DeviceIndex>(-1);

/**
 * @brief Symbolic representation of a tensor dimension size.
 *
 * This class holds an optional size value. It can be in an unspecified state (no value)
 * or a concrete size. Used for shape validation where actual sizes are compared against
 * expected symbolic values.
 */
class SymbolicSize {
 public:
  /**
   * @brief Construct a symbolic size in the unspecified state.
   */
  SymbolicSize() = default;

  /**
   * @brief Check if a concrete size has been set.
   * @return true if a concrete size is stored, false otherwise.
   */
  auto HasValue() const -> bool { return m_size_ != yllang::kAnySize; }

  /**
   * @brief Set the concrete size value.
   * @param size The size value to set.
   * @throws RuntimeCheck if a value is already set.
   */
  auto SetValue(const int64_t &size) -> void {
    RuntimeCheck(!HasValue(), "SymbolicSize::SetValue check failed");
    m_size_ = size;
  }

  /**
   * @brief Get the stored size as an optional.
   * @return std::optional containing the size if HasValue() is true, otherwise std::nullopt.
   */
  auto GetValue() -> std::optional<int64_t> { return HasValue() ? std::optional{m_size_} : std::nullopt; }

  /**
   * @brief Get the concrete size, assuming a value has been set.
   * @return The stored size.
   * @throws RuntimeCheck if no value is set.
   */
  auto UnWrap() const -> int64_t {
    RuntimeCheck(HasValue(), "SymbolicSize::UnWrap check failed");
    return m_size_;
  }

  /**
   * @brief Verify that the given size matches the stored symbolic value.
   *
   * If this symbolic size has no value yet, it sets the value to the given size.
   * Otherwise, it checks that the given size equals the stored value.
   *
   * @param size The actual size to verify.
   * @throws RuntimeCheck if a stored value exists and does not match.
   */
  auto Verify(const int64_t &size) -> void {
    if (HasValue()) {
      RuntimeCheck(size == m_size_, "SymbolicSize::Verify check failed");
    } else {
      SetValue(size);
    }
  }

 private:
  int64_t m_size_{yllang::kAnySize};  ///< Stored size, initially kAnySize (unspecified).
};

/**
 * @brief Symbolic representation of a torch device.
 *
 * Stores an optional device (type and index). Initially in an unspecified state.
 */
class SymbolicDevice {
 public:
  /**
   * @brief Construct a symbolic device in the unspecified state.
   */
  SymbolicDevice() : m_device_(yllang::kNullDevice, yllang::kAnyDeviceId) {}

  /**
   * @brief Check if a concrete device has been set.
   * @return true if a device is stored, false otherwise.
   */
  auto HasValue() const -> bool { return yllang::kNullDevice != m_device_.type(); }

  /**
   * @brief Get the stored device as an optional.
   * @return std::optional containing the device if HasValue() is true, otherwise std::nullopt.
   */
  auto GetValue() const -> std::optional<c10::Device> { return HasValue() ? std::optional{m_device_} : std::nullopt; }

  /**
   * @brief Set the concrete device.
   * @param device The device to store.
   * @throws RuntimeCheck if a device is already set.
   */
  auto SetValue(const c10::Device &device) -> void {
    RuntimeCheck(!HasValue(), "SymbolicDType::SetValue check failed");
    m_device_ = device;
  }

  /**
   * @brief Get the concrete device, assuming a value has been set.
   * @return The stored device.
   * @throws RuntimeCheck if no value is set.
   */
  auto UnWrap() const -> c10::Device {
    RuntimeCheck(HasValue(), "SymbolicDType::UnWrap check failed");
    return m_device_;
  }

  /**
   * @brief Verify that the given device matches the stored symbolic device.
   *
   * If no device is stored, sets it to the given device.
   * Otherwise, checks equality.
   *
   * @param device The actual device to verify.
   * @throws RuntimeCheck if a stored device exists and does not match.
   */
  auto Verify(const c10::Device &device) -> void {
    if (HasValue()) {
      RuntimeCheck(m_device_ == device, "SymbolicSize::Verify check failed");
    } else {
      SetValue(device);
    }
  }

 private:
  c10::Device m_device_;  ///< Stored device, initially null.
};

/**
 * @brief Symbolic representation of a torch data type (dtype).
 *
 * Stores an optional caffe2::TypeMeta. Initially unspecified.
 */
class SymbolicDType {
 public:
  /**
   * @brief Construct a symbolic dtype in the unspecified state.
   */
  SymbolicDType() = default;

  /**
   * @brief Check if a concrete dtype has been set.
   * @return true if a dtype is stored, false otherwise.
   */
  auto HasValue() const -> bool { return caffe2::TypeIdentifier::uninitialized() != m_dtype_.id(); }

  /**
   * @brief Get the stored dtype as an optional.
   * @return std::optional containing the dtype if HasValue() is true, otherwise std::nullopt.
   */
  auto GetValue() const -> std::optional<caffe2::TypeMeta> {
    return HasValue() ? std::optional{m_dtype_} : std::nullopt;
  }

  /**
   * @brief Set the concrete dtype.
   * @param dtype The dtype to store.
   * @throws RuntimeCheck if a dtype is already set.
   */
  auto SetValue(caffe2::TypeMeta dtype) -> void {
    RuntimeCheck(!HasValue(), "SymbolicDType::SetValue check failed");
    m_dtype_ = dtype;
  }

  /**
   * @brief Get the concrete dtype, assuming a value has been set.
   * @return The stored dtype.
   * @throws RuntimeCheck if no value is set.
   */
  auto UnWrap() const -> caffe2::TypeMeta {
    RuntimeCheck(HasValue(), "SymbolicDType::UnWrap check failed");
    return m_dtype_;
  }

  /**
   * @brief Verify that the given dtype matches the stored symbolic dtype.
   *
   * If no dtype is stored, sets it to the given dtype.
   * Otherwise, checks equality.
   *
   * @param dtype The actual dtype to verify.
   * @throws RuntimeCheck if a stored dtype exists and does not match.
   */
  auto Verify(caffe2::TypeMeta dtype) -> void {
    if (HasValue()) {
      RuntimeCheck(dtype == m_dtype_, "SymbolicSize::Verify check failed");
    } else {
      SetValue(dtype);
    }
  }

 private:
  caffe2::TypeMeta m_dtype_;  ///< Stored dtype, initially uninitialized.
};

/**
 * @brief Reference wrapper for SymbolicSize.
 *
 * This class inherits from BaseRef and allows symbolic sizes to be passed by reference.
 * It provides constructors from int64_t (automatically creating a SymbolicSize if needed).
 */
class SymbolicSizeRef : public BaseRef<yllang::SymbolicSize> {
 public:
  /**
   * @brief Default constructor – creates a null reference.
   */
  SymbolicSizeRef() = default;

  /**
   * @brief Construct from an integer size.
   *
   * If the size equals kAnySize, the reference remains null/unspecified.
   * Otherwise, it creates a new SymbolicSize with that value.
   *
   * @param size Concrete size or kAnySize.
   */
  SymbolicSizeRef(int64_t size) {
    if (size != kAnySize) {
      (*this)->SetValue(size);
    } else {
      // Leave unspecified.
    }
  }

  /**
   * @brief Construct by referencing an existing SymbolicSize.
   * @param size Reference to an existing SymbolicSize.
   */
  SymbolicSizeRef(yllang::SymbolicSize &size) : BaseRef<yllang::SymbolicSize>(size) {}
};

/**
 * @brief Reference wrapper for SymbolicDType.
 *
 * Currently empty – used as a placeholder for type-safe reference handling.
 */
class SymbolicDTypeRef : public BaseRef<yllang::SymbolicDType> {
 public:
  // No additional constructors; inherits BaseRef.
};

/**
 * @brief Reference wrapper for SymbolicDevice.
 */
class SymbolicDeviceRef : public BaseRef<yllang::SymbolicDevice> {
 public:
  // No additional constructors; inherits BaseRef.
};

/**
 * @brief A fluent builder and validator for tensor shapes, strides, device and dtype.
 *
 * TensorMatcher allows specifying expected tensor properties symbolically.
 * After construction with an initial shape, you can optionally chain .WithStride(),
 * .WithDType(), .WithDevice() to add expectations. Finally, .Verify() checks a
 * concrete torch::Tensor against these expectations.
 *
 * Example usage:
 *   TensorMatcher({SymbolicSizeRef(2), SymbolicSizeRef(3)})
 *       .WithDType(myDtype)
 *       .WithDevice(myDevice)
 *       .Verify(tensor);
 *
 * The matcher uses references to symbolic objects, so modifications to those objects
 * after building the matcher affect the expectations.
 */
class TensorMatcher {
 public:
  // Disable copy
  TensorMatcher(const TensorMatcher &) = delete;
  auto operator=(const TensorMatcher &) -> TensorMatcher & = delete;

  /**
   * @brief Construct a matcher with an expected shape.
   * @param shape List of symbolic size references, one per dimension.
   */
  explicit TensorMatcher(std::initializer_list<SymbolicSizeRef> shape) : m_shape_(shape) {}

  /**
   * @brief Add stride expectations.
   * @param stride List of symbolic size references for strides, same length as shape.
   * @return Rvalue reference to this matcher for chaining.
   * @throws RuntimeCheck if strides were already set, or if stride size does not match shape.
   */
  auto WithStride(std::initializer_list<SymbolicSizeRef> stride) && -> TensorMatcher && {
    RuntimeCheck(m_strides_.empty(), "");
    RuntimeCheck(stride.size() == m_shape_.size());
    m_strides_ = stride;
    return std::move(*this);
  }

  /**
   * @brief Add dtype expectation.
   * @param dtype Reference to a symbolic dtype.
   * @return Rvalue reference to this matcher for chaining.
   */
  auto WithDType(SymbolicDType &dtype) && -> TensorMatcher && {
    InitType();
    m_dtype_.Rebind(dtype);
    return std::move(*this);
  }

  /**
   * @brief Add device expectation.
   * @param device Reference to a symbolic device.
   * @return Rvalue reference to this matcher for chaining.
   */
  auto WithDevice(SymbolicDevice &device) && -> TensorMatcher && {
    InitDevice();
    m_device_.Rebind(device);
    return std::move(*this);
  }

  /**
   * @brief Verify a concrete tensor against the stored expectations.
   *
   * This function checks:
   * - Tensor dimension matches shape length.
   * - For each dimension, the size matches the corresponding symbolic size (or sets it if unspecified).
   * - If strides were provided, each stride matches the corresponding symbolic stride
   *   (except for dimensions of size 1 where stride is not enforced if symbolic is unspecified).
   * - Dtype matches symbolic dtype (or sets it).
   * - Device matches symbolic device (or sets it).
   *
   * @param view The tensor to verify.
   * @return Rvalue reference to this matcher (allows further chaining).
   * @throws RuntimeCheck on any mismatch.
   */
  auto Verify(const torch::Tensor &view) && -> TensorMatcher && {
    const auto dim = view.dim();
    RuntimeCheck(dim == m_shape_.size());
    for (int64_t i = 0; i < dim; ++i) {
      m_shape_[i]->Verify(view.size(i));
    }
    if (HasStride()) {
      for (int64_t i = 0; i < dim; ++i) {
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
  /**
   * @brief Ensure dtype has not been set before.
   */
  auto InitType() const -> void { RuntimeCheck(!m_dtype_->HasValue(), ""); }
  /**
   * @brief Ensure device has not been set before.
   */
  auto InitDevice() const -> void { RuntimeCheck(!m_device_->HasValue(), ""); }
  /**
   * @brief Check if stride expectations were provided.
   */
  auto HasStride() const -> bool { return !m_strides_.empty(); }

 private:
  std::span<const yllang::SymbolicSizeRef> m_shape_;    ///< Expected shape.
  std::span<const yllang::SymbolicSizeRef> m_strides_;  ///< Expected strides (optional).
  yllang::SymbolicDTypeRef m_dtype_{};                  ///< Expected dtype (optional).
  yllang::SymbolicDeviceRef m_device_{};                ///< Expected device (optional).
};

namespace util {

inline auto SafetensorsType2TorchType(safetensors::dtype dtype) -> torch::ScalarType {
  switch (dtype) {
    case safetensors::dtype::kFLOAT32:
      return torch::kFloat32;
    case safetensors::dtype::kFLOAT64:
      return torch::kFloat64;
    case safetensors::dtype::kFLOAT16:
      return torch::kFloat16;
    case safetensors::dtype::kBFLOAT16:
      return torch::kBFloat16;
    case safetensors::dtype::kINT8:
      return torch::kInt8;
    case safetensors::dtype::kINT16:
      return torch::kInt16;
    case safetensors::dtype::kINT32:
      return torch::kInt32;
    case safetensors::dtype::kINT64:
      return torch::kInt64;
    case safetensors::dtype::kUINT8:
      return torch::kUInt8;
    case safetensors::dtype::kBOOL:
      return torch::kBool;
    default:
      throw std::runtime_error("Unsupported dtype");
  }
}

inline auto ParseTorchDtype(const std::string &dtype_str) -> torch::ScalarType {
  static const std::unordered_map<std::string, torch::ScalarType> dtype_map = {
      {"float32", torch::kFloat32}, {"float16", torch::kFloat16}, {"bfloat16", torch::kBFloat16},
      {"float64", torch::kFloat64}, {"int8", torch::kInt8},       {"int16", torch::kInt16},
      {"int32", torch::kInt32},     {"int64", torch::kInt64},     {"uint8", torch::kUInt8},
      {"bool", torch::kBool},
  };

  auto it = dtype_map.find(dtype_str);
  if (it != dtype_map.end()) {
    return it->second;
  }
  throw std::runtime_error("Unsupported dtype str");
}

/**
 * @brief Copies data from src tensor to dst tensor after verifying shape and dtype.
 *
 * @param dst Target tensor (will be overwritten with src data)
 * @param src Source tensor (must have same shape and dtype as dst)
 *
 * @throws c10::Error via TORCH_CHECK if shape or dtype mismatch.
 */
inline auto CopyTensorWithCheck(torch::Tensor &dst, const torch::Tensor &src) -> void {
  TORCH_CHECK(dst.sizes() == src.sizes(), "Tensor shape mismatch. Expected ", dst.sizes(), ", got ", src.sizes());
  TORCH_CHECK(dst.scalar_type() == src.scalar_type(), "Tensor dtype mismatch. Expected ", dst.scalar_type(), ", got ",
              src.scalar_type());
  dst.copy_(src);
}

}  // namespace util

}  // namespace yllang

#endif  // YLLANG_UTIL_TENSOR_H