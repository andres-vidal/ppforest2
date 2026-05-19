/**
 * @file IO.cpp
 * @brief File I/O utilities, JSON and CSV reading/writing.
 */
#include "io/IO.hpp"
#include "stats/Stats.hpp"
#include "utils/UserError.hpp"

#include <csv.hpp> // IWYU pragma: keep

#include <fmt/format.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>

using namespace ppforest2::types;
using namespace ppforest2::stats;

namespace ppforest2::io {
  void check_file_exists(std::string const& path) {
    user_error(std::filesystem::exists(path), fmt::format("File not found: {}", path));
  }

  void check_file_not_exists(std::string const& path) {
    user_error(!std::filesystem::exists(path), fmt::format("File already exists: {}", path));
  }

  void check_dir_not_exists(std::string const& path) {
    user_error(!std::filesystem::exists(path), fmt::format("Directory already exists: {}", path));
  }
}

namespace ppforest2::io::json {
  std::string ensure_extension(std::string const& path) {
    if (path.size() >= 5 && path.substr(path.size() - 5) == ".json") {
      return path;
    }

    return path + ".json";
  }

  nlohmann::json read_file(std::string const& path, ErrorHandler on_error) {
    std::ifstream in(path);

    on_error(in.is_open(), fmt::format("Could not open file: {}", path));

    try {
      return nlohmann::json::parse(in);
    } catch (nlohmann::json::parse_error const& e) {
      on_error(false, fmt::format("Invalid JSON in file: {}", e.what()));
      throw; // unreachable — on_error always throws
    }
  }

  void write_file(nlohmann::json const& data, std::string const& path, ErrorHandler on_error) {
    std::ofstream out(path);

    on_error(out.is_open(), fmt::format("Could not open file for writing: {}", path));

    out << data.dump(2);
    out.close();
  }
}

namespace ppforest2::io::text {
  void write_file(std::string const& content, std::string const& path, ErrorHandler on_error) {
    std::ofstream out(path);

    on_error(out.is_open(), fmt::format("Could not open file for writing: {}", path));

    out << content;
    out.close();
  }
}

namespace ppforest2::io::csv {
  namespace {
    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    // Signed-int subscript into a `std::vector` (or any container with
    // `operator[]`). Tightens the per-callsite `static_cast<std::size_t>(j)`
    // noise into a single named conversion. Forwards through the proxy
    // reference returned by `std::vector<bool>` correctly.
    template<typename Container> decltype(auto) at(Container&& v, int j) {
      return std::forward<Container>(v)[static_cast<std::size_t>(j)];
    }

    bool is_numeric(std::string const& s) {
      if (s.empty()) {
        return false;
      }

      char* end = nullptr;
      std::strtod(s.c_str(), &end);
      return end != s.c_str() && *end == '\0';
    }

    bool is_floating(std::string const& s) {
      return is_numeric(s) && s.find_first_of(".eE") != std::string::npos;
    }

    struct CategoricalEncoder {
      std::unordered_map<std::string, int> mapping;

      Feature encode(std::string const& value) {
        auto it = mapping.find(value);

        if (it == mapping.end()) {
          int code       = static_cast<int>(mapping.size());
          mapping[value] = code;
          return static_cast<Feature>(code);
        }

        return static_cast<Feature>(it->second);
      }
    };

    // -----------------------------------------------------------------------
    // Parse pipeline
    //
    // CSV → (raw string rows) → (column-type detection + mode detection) →
    // (typed FeatureMatrix / OutcomeVector + group_names). Mode is detected
    // from the y column's written form: any value with fractional /
    // scientific notation (`.`, `e`, `E`) routes to regression (y as float,
    // empty group_names); otherwise y maps to integer codes (classification,
    // with group_names populated). Each phase below is one helper.
    // -----------------------------------------------------------------------

    struct RawRows {
      std::vector<std::vector<std::string>> values;
      int n_cols = 0;
    };

    struct ParsedCSV {
      FeatureMatrix x;
      OutcomeVector y;
      types::Names group_names;
      types::Names feature_names;
    };

    RawRows read_raw_rows(::csv::CSVReader& reader, int expected_cols) {
      RawRows raw;
      raw.n_cols = expected_cols;

      for (::csv::CSVRow& row : reader) {
        int const row_num = static_cast<int>(raw.values.size()) + 1;

        user_error(
            static_cast<int>(row.size()) == raw.n_cols,
            fmt::format("Row {} has {} column(s), expected {} (same as the header)", row_num, row.size(), raw.n_cols)
        );

        std::vector<std::string> values;
        values.reserve(row.size());
        for (auto&& cell : row) {
          values.push_back(cell.get<std::string>());
        }
        raw.values.push_back(std::move(values));
      }

      return raw;
    }

    // A column is categorical if any cell in it fails `is_numeric`.
    std::vector<bool> is_categorical(RawRows const& raw, int n_features) {
      std::vector<bool> res(static_cast<std::size_t>(n_features), false);

      for (int j = 0; j < n_features; ++j) {
        for (auto const& row : raw.values) {
          if (!is_numeric(at(row, j))) {
            at(res, j) = true;
            break;
          }
        }
      }

      return res;
    }

    // Auto-detect regression: any y value with fractional / scientific
    // notation marks y as continuous. Integer-coded labels (e.g. Wine's
    // "1", "2", "3") look numeric but stay on the classification path.
    bool is_classification(RawRows const& raw) {
      int const y_col = raw.n_cols - 1;
      return std::all_of(raw.values.begin(), raw.values.end(), [&](auto const& row) {
        return !is_floating(at(row, y_col));
      });
    }

    Outcome encode_classification_y(
        std::string const& y_str, types::Names& group_names, std::unordered_map<std::string, int>& label_mapping
    ) {
      auto [it, inserted] = label_mapping.try_emplace(y_str, static_cast<int>(group_names.size()));
      if (inserted) {
        group_names.push_back(y_str);
      }
      return static_cast<Outcome>(it->second);
    }

    Outcome parse_regression_y(std::string const& y_str, int row_num) {
      user_error(
          is_numeric(y_str), fmt::format("Row {} response value '{}' is not numeric (regression mode)", row_num, y_str)
      );
      return std::stof(y_str);
    }

    ParsedCSV encode_rows(RawRows const& raw, types::Names feature_names) {
      int const n_features = raw.n_cols - 1;
      int const y_col      = raw.n_cols - 1;
      int const n          = static_cast<int>(raw.values.size());

      auto const _is_classification = is_classification(raw);
      auto const _is_categorical    = is_categorical(raw, n_features);

      std::vector<CategoricalEncoder> encoders(static_cast<std::size_t>(n_features));
      FeatureMatrix x(n, n_features);
      OutcomeVector y(n);
      types::Names group_names;
      std::unordered_map<std::string, int> label_mapping;

      for (int i = 0; i < n; ++i) {
        auto const& row = at(raw.values, i);

        for (int j = 0; j < n_features; ++j) {
          auto const& val = at(row, j);
          x(i, j)         = at(_is_categorical, j) ? at(encoders, j).encode(val) : std::stof(val);
        }

        auto const& y_str = at(row, y_col);

        if (_is_classification) {
          y(i) = encode_classification_y(y_str, group_names, label_mapping);
        } else {
          y(i) = parse_regression_y(y_str, i + 1);
        }
      }

      return {std::move(x), std::move(y), std::move(group_names), std::move(feature_names)};
    }

    ParsedCSV parse_csv(std::string const& filename) {
      ::csv::CSVReader reader(filename);

      auto const col_names = reader.get_col_names();
      user_error(col_names.size() >= 2, "CSV header must have at least 2 columns (features + response)");

      int const n_cols = static_cast<int>(col_names.size());
      types::Names feature_names(col_names.begin(), col_names.end() - 1);

      RawRows raw = read_raw_rows(reader, n_cols);
      user_error(!raw.values.empty(), "CSV file is empty or has no data rows");

      return encode_rows(raw, std::move(feature_names));
    }

    // Feature-only encoding for the `serve` path. No response column, no
    // mode detection, no `group_names`. The raw string rows are moved into
    // the returned `FeatureSet` so callers can pull out non-feature columns
    // (e.g. a response label) without re-parsing the CSV.
    FeatureSet encode_features(RawRows raw, types::Names feature_names) {
      int const n_features = raw.n_cols;
      int const n          = static_cast<int>(raw.values.size());

      auto const _is_categorical = is_categorical(raw, n_features);

      std::vector<CategoricalEncoder> encoders(static_cast<std::size_t>(n_features));
      FeatureMatrix x(n, n_features);

      for (int i = 0; i < n; ++i) {
        auto const& row = at(raw.values, i);

        for (int j = 0; j < n_features; ++j) {
          auto const& val = at(row, j);
          x(i, j)         = at(_is_categorical, j) ? at(encoders, j).encode(val) : std::stof(val);
        }
      }

      return FeatureSet{std::move(x), std::move(feature_names), std::move(raw.values)};
    }

    // Picks the `DataPacket` ctor by mode: classification derives groups
    // from `y` codes (0..G-1); regression has no group concept and passes `{}`.
    DataPacket make_data_packet(ParsedCSV const& parsed) {
      if (parsed.group_names.empty()) {
        return DataPacket(parsed.x, parsed.y, {}, {}, parsed.feature_names);
      } else {
        return DataPacket(parsed.x, parsed.y, parsed.group_names, parsed.feature_names);
      }
    }
  }

  DataPacket read(std::string const& filename) {
    return make_data_packet(parse_csv(filename));
  }

  DataPacket read_sorted(std::string const& filename) {
    try {
      auto parsed = parse_csv(filename);
      sort(parsed.x, parsed.y);
      return make_data_packet(parsed);
    } catch (UserError const&) {
      throw;
    } catch (std::exception const& e) {
      throw UserError(fmt::format("Error reading CSV file '{}': {}", filename, e.what()));
    }
  }

  void write(DataPacket const& data, std::string const& filename) {
    std::ofstream out(filename);

    user_error(out.is_open(), fmt::format("Could not open file for writing: {}", filename));

    Eigen::IOFormat const csv_row(Eigen::StreamPrecision, Eigen::DontAlignCols, ",", ",");

    for (Eigen::Index i = 0; i < data.x.rows(); ++i) {
      out << data.x.row(i).format(csv_row) << "," << data.y[i] << "\n";
    }
  }

  FeatureSet read_features_from_string(std::string const& content) {
    try {
      user_error(!content.empty(), "CSV body is empty");

      ::csv::CSVFormat format;
      format.variable_columns(::csv::VariableColumnPolicy::THROW);

      std::stringstream stream(content);
      ::csv::CSVReader reader(stream, format);

      auto const col_names = reader.get_col_names();
      user_error(!col_names.empty(), "CSV header must have at least 1 column");

      int const n_cols = static_cast<int>(col_names.size());
      types::Names feature_names(col_names.begin(), col_names.end());

      RawRows raw = read_raw_rows(reader, n_cols);
      user_error(!raw.values.empty(), "CSV has no data rows");

      return encode_features(std::move(raw), std::move(feature_names));
    } catch (UserError const&) {
      throw;
    } catch (std::exception const& e) {
      throw UserError(fmt::format("Error parsing CSV: {}", e.what()));
    }
  }
}
