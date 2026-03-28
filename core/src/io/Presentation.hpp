/**
 * @file Presentation.hpp
 * @brief Model statistics, variable importance serialization, and
 *        formatted terminal display for confusion matrices and VI tables.
 */
#pragma once

#include "io/Color.hpp"
#include "io/Output.hpp"
#include "models/VariableImportance.hpp"
#include "stats/ConfusionMatrix.hpp"
#include "stats/Stats.hpp"
#include "utils/Types.hpp"

#include <nlohmann/json.hpp>

namespace ppforest2::io {
  /**
   * @brief Aggregated statistics across multiple training iterations.
   *
   * Stores per-iteration timing and error data, plus the process-wide
   * peak RSS. Provides mean/std accessors and JSON serialization
   * (including a per-iteration breakdown).
   */
  struct ModelStats {
    types::Vector<float> tr_times;
    types::Vector<float> tr_error;
    types::Vector<float> te_error;
    long peak_rss_bytes = -1;

    double mean_time() const {
      return tr_times.mean();
    }

    double mean_tr_error() const {
      return tr_error.mean();
    }

    double mean_te_error() const {
      return te_error.mean();
    }

    double std_time() const {
      return stats::sd(tr_times);
    }

    double std_tr_error() const {
      return stats::sd(tr_error);
    }

    double std_te_error() const {
      return stats::sd(te_error);
    }

    /** @brief Serialize to JSON including per-iteration breakdown. */
    nlohmann::json to_json() const;
  };

  /**
   * @brief Print evaluation results (timing, errors, memory) to stdout.
   * @param stats The aggregated model statistics.
   */
  void print_results(Output& out, const ModelStats& stats);

  /**
   * @brief Print a ranked variable importance table to stdout.
   *
   * Columns: rank, variable name, sigma (scale), VI2 (projections), VI3 (weighted),
   * VI1 (permuted). Rows are sorted by VI2 descending. The VI1 column is
   * omitted when its vector is empty. At most max_rows rows are printed.
   *
   * @param out       Output context.
   * @param vi        Variable importance results.
   * @param max_rows  Maximum number of rows to print (0 = all).
   */
  void print_variable_importance(
    Output&                   out,
    const VariableImportance& vi,
    int                       max_rows = 20);

  /**
   * @brief Print a formatted confusion matrix to stdout.
   *
   * Displays the confusion matrix with group labels, diagonal highlighting,
   * and per-group error rates.
   *
   * @param out Output context.
   * @param cm  The confusion matrix to print.
   */
  void print_confusion_matrix(
    Output&                         out,
    const stats::ConfusionMatrix&   cm,
    const std::string&              title       = "Confusion Matrix",
    const std::vector<std::string>& group_names = {});

  /**
   * @brief Optional display hints for print_configuration.
   *
   * When printing a freshly trained model, callers can provide extra
   * context (default-value tags, vars percentage, train/test split)
   * that isn't stored in the saved config JSON.
   */
  struct ConfigDisplayHints {
    int vars_percent     = -1;
    bool default_vars    = false;
    bool default_threads = false;
    bool default_seed    = false;
    std::string training_samples;
    std::string test_samples;
  };

  /**
   * @brief Print model configuration table from a JSON config object.
   *
   * @param out    Output context.
   * @param config The config JSON (trees, lambda, seed, threads, vars, data).
   * @param hints  Optional display hints for richer output (defaults, percentages).
   */
  void print_configuration(
    Output&                   out,
    const nlohmann::json&     config,
    const ConfigDisplayHints& hints = {});

  /**
   * @brief Print a data summary table from a JSON meta object.
   *
   * Shows observations, features, groups, and group names.
   *
   * @param out  Output context.
   * @param meta The meta JSON (observations, features, groups).
   */
  void print_data_summary(Output& out, const nlohmann::json& meta);

  /**
   * @brief Display a full model summary from its JSON representation.
   *
   * Shows configuration, data summary, training/OOB confusion matrices,
   * degenerate warnings, timing, and variable importance.
   * Used by both `run_train` (after training) and `run_summarize` (from file).
   */
  void print_summary(
    Output&                   out,
    const nlohmann::json&     model_data,
    const ConfigDisplayHints& hints = {});
}
