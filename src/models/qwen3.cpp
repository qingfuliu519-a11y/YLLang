#include "models/qwen3.h"
#include <filesystem>
#include <string>
#include <utility>
#include <vector>
#include "config/config.h"
#include "util/curl_downloader.h"

namespace yllang {

auto Qwen3::ModelFiles() -> std::vector<ModelFileEntry> {
  std::string base_url = "https://hf-mirror.com/Qwen/Qwen3-0.6B-Base/resolve/main/";
  return {
      {base_url + "tokenizer.json", "tokenizer.json"},
      {base_url + "vocab.json", "vocab.json"},
      {base_url + "model.safetensors", "model.safetensors"},
      {base_url + "generation_config.json", "generation_config.json"},
      {base_url + "config.json", "config.json"},
  };
}

auto Qwen3::Load() -> std::unique_ptr<Model> {
  // Build the local model root directory.
  const std::string model_dir = kModelPath + "/" + ModelTypeToString(Qwen3::Kind()) + "/";
  std::filesystem::path dir_path(model_dir);

  // Create the directory if it doesn't exist.
  if (!std::filesystem::exists(dir_path)) {
    if (!std::filesystem::create_directories(dir_path)) {
      return nullptr;  // Failed to create directory.
    }
  }

  // Obtain the list of files that need to be downloaded for this model.
  auto files = ModelFiles();
  if (files.empty()) {
    return nullptr;  // Unsupported model type.
  }

  // Download each file sequentially.
  for (const auto &[remote_url, local_name] : files) {
    std::string local_path = model_dir + local_name;
    if (!util::CurlDownloader::Download(remote_url, local_path)) {
      return nullptr;  // Download failed.
    }
  }

  // After downloading, construct and return a Model object.
  // Actual construction should load the downloaded files as needed.
  auto model = std::make_unique<Qwen3>();
  return model;
}

}  // namespace yllang