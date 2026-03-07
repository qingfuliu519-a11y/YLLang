#ifndef YLLANG_UTIL_DEVICE_H
#define YLLANG_UTIL_DEVICE_H
#include "c10/core/Device.h"
namespace yllang {

using DeviceType = int8_t;
using DeviceIndex = int8_t;

class Device {
 public:
  Device() = default;
  virtual ~Device() = default;

  virtual auto GetDevice() -> DeviceType = 0;
  virtual auto GetDeviceId() -> DeviceIndex = 0;
};

class C10Device final : public Device {
 public:
  C10Device() : m_device_(c10::DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES, -1) {};

  C10Device(const DeviceType &type, const DeviceIndex &index) : m_device_(static_cast<c10::DeviceType>(type), index) {};

  C10Device(const c10::Device &device) : m_device_(device) {};

  ~C10Device() override = default;

  auto GetDevice() -> DeviceType override { return static_cast<DeviceType>(m_device_.type()); }

  auto GetDeviceId() -> DeviceIndex override { return static_cast<DeviceType>(m_device_.index()); }

 private:
  c10::Device m_device_;
};

}  // namespace yllang

#endif  // YLLANG_UTIL_DEVICE_H
