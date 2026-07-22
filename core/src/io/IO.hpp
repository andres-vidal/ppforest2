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
   * @throws UserError on any failure: missing file, empty file, rows whose
   *         column count differs from the header, or a column mixing numeric
   *         and non-numeric cells (stray "NA", empty, or non-finite values).
   */
  stats::DataPacket read(std::string const& filename);

  /**
   * @brief Read a CSV file, parsing the response column in a known mode.
   *
   * Same as `read`, but skips the written-form mode detection: @p mode states
   * how to parse the response column. Classification maps labels to integer
   * codes in first-appearance order (populating `group_names`); regression
   * parses a continuous `float` response and rejects non-numeric values.
   * Rows keep their file order — use this when predictions must line up with
   * the input rows (e.g. `predict`), with the mode taken from the model.
   *
   * @throws UserError on any failure (missing file, parse error, malformed
   *         shape, non-numeric regression response).
   */
  stats::DataPacket read(std::string const& filename, types::Mode mode);

  /**
   * @brief Re-encode `data.y` label codes into a target label space.
   *
   * `data.y` carries integer codes indexing `data.group_names` — the order
   * labels first appeared in the file. Model predictions and metrics use the
   * model's own label order (`meta.groups`), which can differ when a
   * prediction file lists classes in a different order than the training
   * file. Returns a copy of `data.y` re-encoded so that code `i` means
   * `target_groups[i]`.
   *
   * @throws UserError if the file contains a label that is not present in
   *         @p target_groups.
   */
  types::OutcomeVector remap_labels(stats::DataPacket const& data, types::Names const& target_groups);

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
   * @brief Read a CSV file sorted by response, parsing the response in a
   *        known mode.
   *
   * Same as the one-argument `read_sorted`, but skips the written-form mode
   * detection: @p mode states how to parse the response column. Use this
   * when the mode is already decided (e.g. `train --mode regression`) so
   * that integer-written regression responses stay numeric instead of being
   * label-encoded.
   *
   * @throws UserError on any failure, including a non-numeric response in
   *         regression mode.
   */
  stats::DataPacket read_sorted(std::string const& filename, types::Mode mode);

  /**
   * @brief Write a DataPacket to a CSV file (features followed by label, no header).
   * @param data The DataPacket to write.
   * @param filename Output file path.
   */
  void write(stats::DataPacket const& data, std::string const& filename);

  /**
   * @brief Result of parsing a feature-only CSV (no response column).
   *
   * `raw_rows[i][j]` is the original text of the j-th column on row i. It is
   * kept around so callers can inspect columns that aren't part of the model
   * (e.g. a response column carried over from training data, used to compute
   * metrics on /predict).
   */
  struct FeatureSet {
    types::FeatureMatrix x;
    types::Names feature_names;
    std::vector<std::vector<std::string>> raw_rows;
  };

  /**
   * @brief Parse a feature-only CSV from an in-memory string.
   *
   * The first row is the header (feature names); every subsequent row is one
   * observation. There is no response column. Used by the `serve` subcommand
   * to parse `POST /predict` request bodies — categorical encoding runs
   * per-call, so callers must encode categoricals consistently with the
   * training data.
   *
   * @throws UserError on empty body, missing header, no data rows, or
   *         malformed shape.
   */
  FeatureSet read_features_from_string(std::string const& content);
}
