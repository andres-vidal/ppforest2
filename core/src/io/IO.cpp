/**
 * @file IO.cpp
 * @brief File I/O utilities, JSON and CSV reading/writing.
 */
#include "io/IO.hpp"
#include "stats/GroupPartition.hpp"
#include "stats/Stats.hpp"
#include "utils/Invariant.hpp"
#include "utils/UserError.hpp"

#include "csv.hpp"

#include <fmt/format.h>

#include <vector>
#include <unordered_map>
#include <fstream>
#include <filesystem>

namespace ppforest2::io {
  void check_file_not_exists(const std::string& path) {
    if (std::filesystem::exists(path)) {
      fmt::print(stderr, "Error: File already exists: {}\n", path);
      std::exit(1);
    }
  }

  void check_dir_not_exists(const std::string& path) {
    if (std::filesystem::exists(path)) {
      fmt::print(stderr, "Error: Directory already exists: {}\n", path);
      std::exit(1);
    }
  }
}

namespace ppforest2::io::json {
  std::string ensure_extension(const std::string& path) {
    if (path.size() >= 5 && path.substr(path.size() - 5) == ".json") {
      return path;
    }

    return path + ".json";
  }

  nlohmann::json read_file(const std::string& path) {
    std::ifstream in(path);

    if (!in.is_open()) {
      fmt::print(stderr, "Error: Could not open file: {}\n", path);
      std::exit(1);
    }

    try {
      return nlohmann::json::parse(in);
    } catch (const nlohmann::json::parse_error& e) {
      fmt::print(stderr, "Error: Invalid JSON in file: {}\n", e.what());
      std::exit(1);
    }
  }

  void write_file(const nlohmann::json& data, const std::string& path) {
    std::ofstream out(path);

    if (!out.is_open()) {
      fmt::print(stderr, "Error: Could not open file for writing: {}\n", path);
      std::exit(1);
    }

    out << data.dump(2);
    out.close();
  }
}

namespace ppforest2::io::csv {
namespace {
  bool is_numeric(const std::string& s) {
    if (s.empty()) return false;

    char *end = nullptr;
    std::strtod(s.c_str(), &end);
    return end != s.c_str() && *end == '\0';
  }

  struct CategoricalEncoder {
    std::unordered_map<std::string, int> mapping;

    types::Feature encode(const std::string& value) {
      auto it = mapping.find(value);

      if (it == mapping.end()) {
        int code = static_cast<int>(mapping.size());
        mapping[value] = code;
        return static_cast<types::Feature>(code);
      }

      return static_cast<types::Feature>(it->second);
    }
  };
}

  stats::DataPacket read(const std::string& filename) {
    ::csv::CSVReader reader(filename);

    // Extract feature column names from header (all columns except the last).
    auto col_names = reader.get_col_names();
    std::vector<std::string> feature_names;

    if (!col_names.empty()) {
      feature_names.assign(col_names.begin(), col_names.end() - 1);
    }

    // First pass: read all raw string values to detect categorical columns.
    std::vector<std::vector<std::string>> raw_rows;
    int n_cols = 0;

    for (::csv::CSVRow& row : reader) {
      int row_num = static_cast<int>(raw_rows.size()) + 1;
      ppforest2::user_error(row.size() >= 2, fmt::format("Row {} has only {} column(s) — expected at least 2 (features + label)", row_num, row.size()));

      std::vector<std::string> values;

      for (int j = 0; j < row.size(); ++j) {
        values.push_back(row[j].get<std::string>());
      }

      if (n_cols == 0) {
        n_cols = static_cast<int>(values.size());
      }

      raw_rows.push_back(std::move(values));
    }

    ppforest2::user_error(!raw_rows.empty(), "CSV file is empty or has no data rows");

    const int n_features = n_cols - 1;

    // Detect which feature columns are categorical (non-numeric).
    std::vector<bool> is_categorical(static_cast<std::size_t>(n_features), false);

    for (int j = 0; j < n_features; ++j) {
      for (const auto& row : raw_rows) {
        if (!is_numeric(row[static_cast<std::size_t>(j)])) {
          is_categorical[static_cast<std::size_t>(j)] = true;
          break;
        }
      }
    }

    // Encode features (numeric or categorical) and response labels.
    std::vector<CategoricalEncoder> encoders(static_cast<std::size_t>(n_features));

    const int n = static_cast<int>(raw_rows.size());

    types::FeatureMatrix x(n, n_features);
    types::ResponseVector y(n);

    std::unordered_map<std::string, int> label_mapping;
    std::vector<std::string> label_names;

    for (int i = 0; i < n; ++i) {
      const auto& row = raw_rows[static_cast<std::size_t>(i)];
      ppforest2::user_error(static_cast<int>(row.size()) == n_cols, fmt::format("Row {} has {} column(s), expected {} (same as row 1)", i + 1, row.size(), n_cols));

      for (int j = 0; j < n_features; ++j) {
        const auto& val = row[static_cast<std::size_t>(j)];

        if (is_categorical[static_cast<std::size_t>(j)]) {
          x(i, j) = encoders[static_cast<std::size_t>(j)].encode(val);
        } else {
          x(i, j) = std::stof(val);
        }
      }

      // Response label (last column).
      const std::string& label_str = row[static_cast<std::size_t>(n_cols - 1)];

      if (label_mapping.find(label_str) == label_mapping.end()) {
        label_mapping[label_str] = static_cast<int>(label_names.size());
        label_names.push_back(label_str);
      }

      y[i] = label_mapping[label_str];
    }

    return stats::DataPacket(x, y, label_names, feature_names);
  }

  stats::DataPacket read_sorted(const std::string& filename) {
    stats::DataPacket data = read(filename);

    types::FeatureMatrix x  = data.x;
    types::ResponseVector y = data.y;

    if (!stats::GroupPartition::is_contiguous(y)) {
      stats::sort(x, y);
    }

    return stats::DataPacket(x, y, data.group_names, data.feature_names);
  }

  void write(const stats::DataPacket& data, const std::string& filename) {
    std::ofstream out(filename);

    if (!out.is_open()) {
      fmt::print(stderr, "Error: Could not open file for writing: {}\n", filename);
      std::exit(1);
    }

    for (int i = 0; i < data.x.rows(); ++i) {
      for (int j = 0; j < data.x.cols(); ++j) {
        out << data.x(i, j);

        if (j < data.x.cols() - 1) {
          out << ",";
        }
      }

      out << "," << data.y[i] << "\n";
    }

    out.close();
  }
}
