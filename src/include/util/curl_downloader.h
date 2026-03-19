#ifndef YLLANG_UTIL_CURL_DOWNLOADER_H
#define YLLANG_UTIL_CURL_DOWNLOADER_H

#include <curl/curl.h>
#include <fstream>
#include <string>
#include "util/singleton.h"

namespace yllang {
namespace util {

/**
 * @brief A singleton HTTP/HTTPS downloader using libcurl.
 *
 * This class provides a convenient interface for downloading files from URLs
 * with progress display and proxy support. It is implemented as a singleton
 * to ensure libcurl global initialization is done only once.
 */
class CurlDownloader : public Singleton<CurlDownloader> {
  friend class Singleton;

 public:
  CurlDownloader(const CurlDownloader &) = delete;
  CurlDownloader &operator=(const CurlDownloader &) = delete;
  CurlDownloader(CurlDownloader &&) = delete;
  CurlDownloader &operator=(CurlDownloader &&) = delete;

  /**
   * @brief Downloads a file from the given URL to the specified local path.
   *
   * Displays a progress bar during download and automatically handles
   * redirects, HTTPS, proxy settings (from HTTP_PROXY/HTTPS_PROXY env vars),
   * and Hugging Face authentication (HF_TOKEN env var).
   *
   * @param url         The source URL (HTTP/HTTPS).
   * @param output_path Local file path where the downloaded content will be saved.
   * @return bool       True if download succeeded and HTTP status code is 200, false otherwise.
   */
  static auto Download(const std::string &url, const std::string &output_path) -> bool;

 private:
  CurlDownloader();
  ~CurlDownloader() override;
};

}  // namespace util
}  // namespace yllang

#endif  // YLLANG_UTIL_CURL_DOWNLOADER_H