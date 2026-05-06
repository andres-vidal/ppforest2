/**
 * @file TempFile.hpp
 * @brief RAII temporary file and directory with automatic cleanup.
 */
#pragma once

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <io.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace ppforest2::io {
  /**
   * @brief RAII temporary file with automatic cleanup.
   *
   * Creates a unique temporary file on construction and deletes it in
   * the destructor.  Provides helpers to read content back or clear()
   * the file so the path can be used as a fresh output target.
   */
  class TempFile {
  public:
    explicit TempFile(std::string const& suffix = ".json") {
      // clang-format off
      #ifdef _WIN32
      char tmp_dir[MAX_PATH];
      GetTempPathA(MAX_PATH, tmp_dir);
      char tmp_file[MAX_PATH];
      GetTempFileNameA(tmp_dir, "ppforest2", 0, tmp_file);
      sentinel_ = tmp_file;
      path_ = sentinel_ + suffix;
      std::ofstream touch(path_);
      #else
      std::string tmpl = "/tmp/ppforest2_XXXXXX" + suffix;
      std::vector<char> tmpl_buf(tmpl.begin(), tmpl.end());
      tmpl_buf.push_back('\0');

      int fd = mkstemps(tmpl_buf.data(), static_cast<int>(suffix.size()));

      if (fd != -1) {
        path_ = tmpl_buf.data();
        close(fd);
      }
      #endif
      // clang-format on
    }

    ~TempFile() {
      if (!path_.empty()) {
        std::remove(path_.c_str());
      }

      // clang-format off
      #ifdef _WIN32
      if (!sentinel_.empty()) {
        std::remove(sentinel_.c_str());
      }
      #endif
      // clang-format on
    }

    TempFile(TempFile const&)            = delete;
    TempFile& operator=(TempFile const&) = delete;

    TempFile(TempFile&& other) noexcept
        : path_(std::move(other.path_)) {
      other.path_.clear();
      // clang-format off
      #ifdef _WIN32
      sentinel_ = std::move(other.sentinel_);
      other.sentinel_.clear();
      #endif
      // clang-format on
    }

    TempFile& operator=(TempFile&& other) noexcept {
      if (this != &other) {
        if (!path_.empty()) {
          std::remove(path_.c_str());
        }

        // clang-format off
        #ifdef _WIN32
        if (!sentinel_.empty()) {
          std::remove(sentinel_.c_str());
        }
        sentinel_ = std::move(other.sentinel_);
        other.sentinel_.clear();
        #endif
        // clang-format on

        path_ = std::move(other.path_);
        other.path_.clear();
      }

      return *this;
    }

    std::string const& path() const { return path_; }

    /** @brief Remove the file so the path can be used as a fresh output target. */
    void clear() const { std::remove(path_.c_str()); }

    /** @brief Read the entire file contents as a string. */
    std::string read() const {
      std::ifstream in(path_);
      std::stringstream ss;
      ss << in.rdbuf();
      return ss.str();
    }

  private:
    std::string path_;
    // clang-format off
    #ifdef _WIN32
    std::string sentinel_;  // keep the .tmp file alive to prevent GetTempFileNameA reuse
    #endif
    // clang-format on
  };

  /**
   * @brief RAII temporary directory with automatic cleanup.
   *
   * Creates a unique temporary directory and recursively removes it
   * in the destructor.
   */
  class TempDir {
  public:
    TempDir() {
      // clang-format off
      #ifdef _WIN32
      // clang-format on
      char tmp_dir[MAX_PATH];
      GetTempPathA(MAX_PATH, tmp_dir);
      char tmp_file[MAX_PATH];
      GetTempFileNameA(tmp_dir, "ppd", 0, tmp_file);
      // GetTempFileNameA creates a file; replace it with a directory
      std::remove(tmp_file);
      path_ = tmp_file;
      std::filesystem::create_directories(path_);
      // clang-format off
      #else
      // clang-format on
      path_ = "/tmp/ppforest2_dir_XXXXXX";
      std::vector<char> buf(path_.begin(), path_.end());
      buf.push_back('\0');
      char* result = mkdtemp(buf.data());

      if (result != nullptr) {
        path_ = result;
      }
      // clang-format off
      #endif
      // clang-format on
    }

    ~TempDir() {
      if (!path_.empty()) {
        std::filesystem::remove_all(path_);
      }
    }

    TempDir(TempDir const&)            = delete;
    TempDir& operator=(TempDir const&) = delete;

    std::string const& path() const { return path_; }

    /** @brief Return a path inside this directory (file need not exist yet). */
    std::string file(std::string const& name) const { return (std::filesystem::path(path_) / name).string(); }

  private:
    std::string path_;
  };
}
