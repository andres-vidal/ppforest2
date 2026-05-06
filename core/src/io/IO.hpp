/**
 * @file IO.hpp
 * @brief File I/O utilities, JSON and CSV reading/writing.
 */
#pragma once

#include "stats/DataPacket.hpp"

#include "utils/Invariant.hpp"

#include <nlohmann/json.hpp>
#include <string>

namespace ppforest2::io {
  using ErrorHandler = void (*)(bool, std::string const&);

  /**
   * @brief Exit with an error if a file does not exist at the given path.
   * @param path The file path to check.
   */
  void check_file_exists(std::string const& path);

  /**
   * @brief Exit with an error if a file already exists at the given path.
   * @param path The file path to check.
   */
  void check_file_not_exists(std::string const& path);

  /**
   * @brief Exit with an error if a directory already exists at the given path.
   * @param path The directory path to check.
   */
  void check_dir_not_exists(std::string const& path);
}

namespace ppforest2::io::json {
  /**
   * @brief Ensure a file path ends with the ".json" extension.
   * @param path The original file path.
   * @return The path with ".json" appended if it was missing.
   */
  std::string ensure_extension(std::string const& path);

  /**
   * @brief Read a JSON file and parse its contents.
   * @param path    The input file path.
   * @param on_error Error handler for file/parse failures.
   *                 Defaults to invariant(); pass user_error for
   *                 user-provided paths.
   * @return The parsed JSON object.
   */
  nlohmann::json read_file(std::string const& path, ErrorHandler on_error = invariant);

  /**
   * @brief Write a JSON object to a file (pretty-printed with indent 2).
   * @param data     The JSON object to serialize.
   * @param path     The output file path.
   * @param on_error Error handler for file failures.
   *                 Defaults to invariant(); pass user_error for
   *                 user-provided paths.
   */
  void write_file(nlohmann::json const& data, std::string const& path, ErrorHandler on_error = invariant);
}

namespace ppforest2::io::text {
  /**
   * @brief Write a string to a file.
   * @param content  The text to write.
   * @param path     The output file path.
   * @param on_error Error handler for file failures.
   */
  void write_file(std::string const& content, std::string const& path, ErrorHandler on_error = invariant);
}

namespace ppforest2::io::csv {
  /**
   * @brief Read a CSV file into a DataPacket.
   *
   * Assumes the last column is the response variable (group label as string)
   * and all preceding columns are features. Categorical feature columns are
   * automatically detected and integer-encoded. String labels are mapped
   * to contiguous integer codes starting at 0.
   *
   * @param filename Path to the CSV file.
   * @return A DataPacket containing the feature matrix and response vector.
   * @throws std::runtime_error If the file is empty or has inconsistent columns.
   */
  stats::DataPacket read(std::string const& filename);

  /**
   * @brief Read a CSV file and sort rows ascending by the response column.
   *
   * Mode is detected from the y column's *written form*:
   * - If any value carries fractional or scientific notation (`.`, `e`, `E`),
   *   y is parsed as a continuous `float` response (regression shape) and
   *   `group_names` is empty.
   * - Otherwise, y is mapped to integer codes in first-appearance order and
   *   `group_names` carries the original label strings (classification
   *   shape). This keeps integer-coded label CSVs like Wine and Glass on
   *   the classification path.
   *
   * Rows are sorted ascending by the encoded y, which gives the training
   * routines what they need: classification — contiguous groups; regression —
   * y-ordered rows for `ByCutpoint::init`'s median split.
   *
   * @throws UserError on any failure (missing file, parse error, malformed
   *         shape) — CSV reading failures are user-facing by nature.
   */
  stats::DataPacket read_sorted(std::string const& filename);

  /**
   * @brief Write a DataPacket to a CSV file (features followed by label, no header).
   * @param data The DataPacket to write.
   * @param filename Output file path.
   */
  void write(stats::DataPacket const& data, std::string const& filename);
}
