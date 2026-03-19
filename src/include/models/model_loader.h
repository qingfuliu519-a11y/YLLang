/**
 * @file model_loader.h
 * @brief Defines a generic model loader with compile-time registration and type safety.
 */

#ifndef YLLANG_MODEL_LOADER_H
#define YLLANG_MODEL_LOADER_H

#include <functional>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include "models/model.h"

namespace yllang {

// ---------- Helper macro for generating static method detection ----------

#define DEFINE_HAS_STATIC_METHOD(MethodName, ReturnType)               \
  template <typename T, typename = void>                               \
  struct Has##MethodName : std::false_type {};                         \
                                                                       \
  template <typename T>                                                \
  struct Has##MethodName<T, std::void_t<decltype(T::MethodName())>>    \
      : std::is_convertible<decltype(T::MethodName()), ReturnType> {}; \
                                                                       \
  template <typename T>                                                \
  inline constexpr bool Has##MethodName##V = Has##MethodName<T>::value

// ---------- Generate type traits for Load and Kind ----------
DEFINE_HAS_STATIC_METHOD(Load, std::unique_ptr<Model>);
DEFINE_HAS_STATIC_METHOD(Kind, ModelType);

// ---------- Factory class ----------
class ModelLoader {
 public:
  using ModelCreator = std::function<std::unique_ptr<Model>()>;

  static void RegisterModel(ModelType model_type, ModelCreator model_creator) {
    GetRegistry()[model_type] = std::move(model_creator);
  }

  static std::unique_ptr<Model> Load(ModelType model_type) {
    auto &registry = GetRegistry();
    auto it = registry.find(model_type);
    if (it != registry.end()) {
      return it->second();
    }
    return nullptr;
  }

 private:
  static std::unordered_map<ModelType, ModelCreator> &GetRegistry() {
    static std::unordered_map<ModelType, ModelCreator> registry;
    return registry;
  }
};

// ---------- Helper for automatic registration ----------
template <typename T>
class RegisterHelper {
  static_assert(HasLoadV<T>, "Error: T must have a static Load() function that returns std::unique_ptr<Model>");
  static_assert(HasKindV<T>, "Error: T must have a static Kind() function that returns ModelType");

 public:
  RegisterHelper() {
    ModelLoader::RegisterModel(T::Kind(), []() -> std::unique_ptr<Model> { return T::Load(); });
  }
};

}  // namespace yllang

#endif  // YLLANG_MODEL_LOADER_H