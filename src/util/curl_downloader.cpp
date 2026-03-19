#include "util/curl_downloader.h"
#include <curl/curl.h>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace yllang::util {

CurlDownloader::CurlDownloader() { curl_global_init(CURL_GLOBAL_DEFAULT); }
CurlDownloader::~CurlDownloader() { curl_global_cleanup(); }

/**
 * @brief Formats a size in bytes into a human-readable string (e.g., "1.23 MB").
 *
 * @param bytes The size in bytes.
 * @return std::string Formatted string with appropriate unit.
 */
static auto FormatSize(curl_off_t bytes) -> std::string {
  const char *units[] = {"B", "KB", "MB", "GB", "TB"};
  int unit_idx = 0;
  auto size = static_cast<double>(bytes);
  while (size >= 1024.0 && unit_idx < 4) {
    size /= 1024.0;
    ++unit_idx;
  }
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2) << size << " " << units[unit_idx];
  return oss.str();
}

/**
 * @brief Formats a time duration in seconds into a human-readable string.
 *
 * Formats as "Xs", "XmYs", or "XhYm" depending on the duration.
 *
 * @param seconds Time in seconds.
 * @return std::string Formatted string.
 */
static auto FormatTime(double seconds) -> std::string {
  if (seconds < 60) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(0) << seconds << "s";
    return oss.str();
  }
  if (seconds < 3600) {
    int mins = static_cast<int>(seconds / 60);
    int secs = static_cast<int>(seconds) % 60;
    std::ostringstream oss;
    oss << mins << "m" << std::setw(2) << std::setfill('0') << secs << "s";
    return oss.str();
  }

  int hours = static_cast<int>(seconds / 3600);
  int mins = (static_cast<int>(seconds) % 3600) / 60;
  std::ostringstream oss;
  oss << hours << "h" << std::setw(2) << std::setfill('0') << mins << "m";
  return oss.str();
}

struct ProgressInfo {
  std::string m_filename_;
  std::ostream *m_output_;
  std::chrono::steady_clock::time_point m_start_time_;
  std::chrono::steady_clock::time_point m_last_time_;
  curl_off_t m_last_dlnow_;
  bool m_first_call_;
  int m_spinner_idx_;
};

/**
 * @brief libcurl write callback that writes received data to an output file stream.
 *
 * @param ptr   Pointer to received data.
 * @param size  Size of each data element.
 * @param nmemb Number of elements.
 * @param stream User-supplied pointer (points to the output std::ofstream).
 * @return size_t Number of bytes processed (size * nmemb).
 */
static auto WriteDataCallback(void *ptr, size_t size, size_t nmemb, void *stream) -> size_t {
  auto *out = static_cast<std::ofstream *>(stream);
  out->write(static_cast<char *>(ptr), size * nmemb);
  return size * nmemb;
}

/**
 * @brief libcurl progress callback that displays a download progress bar and speed information.
 *
 * It uses the provided ProgressInfo structure to track download statistics and updates the
 * console with a dynamic progress bar. The callback is invoked periodically by libcurl.
 *
 * @param clientp User-supplied pointer (points to a ProgressInfo struct).
 * @param dltotal Total bytes to download (0 if unknown).
 * @param dlnow   Bytes downloaded so far.
 * @param ultotal Total bytes to upload (unused).
 * @param ulnow   Bytes uploaded so far (unused).
 * @return int 0 to continue, non‑zero to abort.
 */
static auto ProgressCallback(void *clientp, curl_off_t dltotal, curl_off_t dlnow, curl_off_t ultotal, curl_off_t ulnow)
    -> int {
  auto *info = static_cast<ProgressInfo *>(clientp);
  if (nullptr == info || nullptr == info->m_output_) {
    return 0;
  }

  if (info->m_first_call_) {
    info->m_start_time_ = std::chrono::steady_clock::now();
    info->m_last_time_ = info->m_start_time_;
    info->m_last_dlnow_ = 0;
    info->m_first_call_ = false;
    info->m_spinner_idx_ = 0;
  }

  auto now = std::chrono::steady_clock::now();
  double elapsed_total = std::chrono::duration<double>(now - info->m_start_time_).count();
  double elapsed_since_last = std::chrono::duration<double>(now - info->m_last_time_).count();
  curl_off_t delta = dlnow - info->m_last_dlnow_;
  double speed_inst = (elapsed_since_last > 0.001) ? (delta / elapsed_since_last) : 0.0;
  double speed_avg = (elapsed_total > 0.001) ? (dlnow / elapsed_total) : 0.0;

  info->m_last_time_ = now;
  info->m_last_dlnow_ = dlnow;

  const int bar_width = 40;
  std::ostringstream line;

  if (dltotal <= 0) {
    const char spinner[] = {'|', '/', '-', '\\'};
    char spin = spinner[info->m_spinner_idx_ % 4];
    info->m_spinner_idx_++;

    line << "\r\033[36m" << info->m_filename_ << "\033[0m: \033[33m" << spin << "\033[0m [";
    for (int i = 0; i < bar_width; ++i) {
      if (i == (info->m_spinner_idx_ % bar_width)) {
        line << "\033[32m>\033[0m";
      } else {
        line << "\033[90m-\033[0m";
      }
    }
    line << "] \033[32m" << FormatSize(dlnow) << "\033[0m downloaded";
    if (speed_inst > 0) {
      line << " (\033[35m" << FormatSize(static_cast<curl_off_t>(speed_inst)) << "/s\033[0m)";
    }
    *info->m_output_ << line.str() << std::flush;
    return 0;
  }

  double percent = static_cast<double>(dlnow) / dltotal * 100.0;
  int filled = static_cast<int>(bar_width * dlnow / dltotal);
  if (filled > bar_width) {
    filled = bar_width;
  }

  double eta = 0.0;
  if (speed_avg > 0) {
    eta = (dltotal - dlnow) / speed_avg;
  }

  line << "\r\033[36m" << info->m_filename_ << "\033[0m: [";
  for (int i = 0; i < bar_width; ++i) {
    if (i < filled) {
      if (i == filled - 1) {
        line << "\033[32m>\033[0m";
      } else {
        line << "\033[32m=\033[0m";
      }
    } else {
      line << "\033[90m-\033[0m";
    }
  }
  line << "] \033[32m" << std::fixed << std::setprecision(1) << percent << "%\033[0m  ";
  line << "\033[32m" << FormatSize(dlnow) << "\033[0m / \033[32m" << FormatSize(dltotal) << "\033[0m";
  if (speed_inst > 0) {
    line << " (\033[35m" << FormatSize(static_cast<curl_off_t>(speed_inst)) << "/s\033[0m)";
  }
  if (eta > 0 && eta < 86400) {
    line << " \033[33mETA: " << FormatTime(eta) << "\033[0m";
  }

  *info->m_output_ << line.str() << std::flush;
  return 0;
}

auto CurlDownloader::Download(const std::string &url, const std::string &output_path) -> bool {
  CURL *curl = curl_easy_init();
  if (nullptr == curl) {
    return false;
  }

  std::ofstream outfile(output_path, std::ios::binary);
  if (!outfile.is_open()) {
    curl_easy_cleanup(curl);
    return false;
  }

  std::string filename = output_path.substr(output_path.find_last_of("/\\") + 1);
  if (filename.empty()) {
    filename = "download";
  }

  ProgressInfo progress_info{
      filename, &std::cout, std::chrono::steady_clock::now(), std::chrono::steady_clock::now(), 0, true, 0};

  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteDataCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &outfile);
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(curl, CURLOPT_FAILONERROR, 0L);
  curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 10L);
  curl_easy_setopt(curl, CURLOPT_USERAGENT, "Mozilla/5.0 (compatible; MyDownloader/1.0)");
  curl_easy_setopt(curl, CURLOPT_VERBOSE, 0L);

  curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);

  curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
  curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);
  curl_easy_setopt(curl, CURLOPT_SSLVERSION, CURL_SSLVERSION_TLSv1_2);

  const char *http_proxy = std::getenv("HTTP_PROXY");
  const char *https_proxy = std::getenv("HTTPS_PROXY");
  if (nullptr != https_proxy && https_proxy[0] != '\0') {
    curl_easy_setopt(curl, CURLOPT_PROXY, https_proxy);
  } else if (nullptr != http_proxy && http_proxy[0] != '\0') {
    curl_easy_setopt(curl, CURLOPT_PROXY, http_proxy);
  }

  struct curl_slist *headers = nullptr;
  const char *hf_token = std::getenv("HF_TOKEN");
  if (nullptr != hf_token && hf_token[0] != '\0') {
    std::string auth_header = "Authorization: Bearer " + std::string(hf_token);
    headers = curl_slist_append(headers, auth_header.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  }

  curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
  curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, ProgressCallback);
  curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &progress_info);

  CURLcode res = curl_easy_perform(curl);
  bool success = false;
  int64_t http_code = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

  if (res == CURLE_OK) {
    if (http_code == 200) {
      success = true;
      std::cout << "\r\033[32m✓\033[0m " << filename << ": Download completed (\033[32m"
                << FormatSize(progress_info.m_last_dlnow_) << "\033[0m)  \n";
    } else {
      std::cerr << "\n\033[31mHTTP error: " << http_code << "\033[0m\n";
    }
  } else {
    std::cerr << "\n\033[31mcurl error: " << curl_easy_strerror(res) << "\033[0m\n";
    if (http_code > 0) {
      std::cerr << "HTTP status code: " << http_code << std::endl;
    }
    if (res == CURLE_SSL_CONNECT_ERROR || res == CURLE_SSL_CERTPROBLEM || res == CURLE_SSL_CIPHER ||
        res == CURLE_SSL_ENGINE_SETFAILED) {
      std::cerr << "\033[33mTLS/SSL 握手失败。可能原因：网络代理、证书问题或 TLS 版本不兼容。"
                << "请检查代理设置或尝试禁用证书验证（测试用）。\033[0m\n";
    }
  }

  if (nullptr != headers) {
    curl_slist_free_all(headers);
  }
  outfile.close();
  curl_easy_cleanup(curl);
  return success;
}

}  // namespace yllang::util