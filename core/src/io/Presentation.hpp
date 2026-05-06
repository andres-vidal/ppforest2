/**
 * @file Presentation.hpp
 * @brief Model statistics, variable importance serialization, and
 *        formatted terminal display for confusion matrices and VI tables.
 */
#pragma once

#include "io/EvaluateResult.hpp"
#include "io/Output.hpp"
#include "models/Evaluation.hpp"
#include "stats/ConfusionMatrix.hpp"
#include "stats/Metrics.hpp"

#include <nlohmann/json.hpp>

namespace ppforest2::io {

  /**
   * @brief Print evaluation results (timing, errors, memory) to stdout.
   * @param stats The aggregated model statistics.
   */
  void print_results(Output& out, ModelStats const& stats);

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
      Output& out, VariableImportance const& vi, types::Names const& feature_names = {}, int max_rows = 20
  );

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
      Output& out,
      stats::ConfusionMatrix const& cm,
      std::string const& title        = "Confusion Matrix",
      types::Names const& group_names = {}
  );

  /**
   * @brief Print regression metrics to stdout.
   *
   * Displays MSE, MAE, and R-squared.
   *
   * @param out     Output context.
   * @param rm      The regression metrics to print.
   * @param title   Section title.
   */
  void print_regression_metrics(
      Output& out, stats::RegressionMetrics const& rm, std::string const& title = "Regression Metrics"
  );

  /**
   * @brief Print a labeled metrics block — error rate + confusion matrix
   * for classification, MSE + regression metrics for regression.
   *
   * The variant overload dispatches on the alternative; callers that already
   * carry a `stats::Metrics` (predict, summarize, OOB) can stay mode-agnostic.
   * Label-less overloads emit unprefixed titles (e.g. "Error:" instead of
   * "Training Error:") for callers that have no section context to add.
   *
   * @param out         Output context.
   * @param metrics     Mode-specific or variant-typed metrics.
   * @param label       Prefix used on row labels (e.g. "Training", "OOB").
   * @param group_names Class labels — used only by the classification path.
   */
  void print_metrics_block(
      Output& out,
      stats::ClassificationMetrics const& cm,
      std::string const& label,
      types::Names const& group_names = {}
  );
  void print_metrics_block(
      Output& out, stats::RegressionMetrics const& rm, std::string const& label, types::Names const& group_names = {}
  );
  void print_metrics_block(
      Output& out, stats::Metrics const& metrics, std::string const& label, types::Names const& group_names = {}
  );

  /** @copydoc print_metrics_block */
  void print_metrics_block(Output& out, stats::ClassificationMetrics const& m, types::Names const& group_names = {});
  void print_metrics_block(Output& out, stats::RegressionMetrics const& m, types::Names const& group_names = {});
  void print_metrics_block(Output& out, stats::Metrics const& m, types::Names const& group_names = {});

  /**
   * @brief Optional display hints for print_configuration.
   *
   * When printing a freshly trained model, callers can provide extra
   * context (default-value tags, vars percentage, train/test split)
   * that isn't stored in the saved config JSON.
   */
  struct ConfigDisplayHints {
    float vars_percent   = -1;
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
  void print_configuration(Output& out, nlohmann::json const& config, ConfigDisplayHints const& hints = {});

  /**
   * @brief Print a data summary table from a JSON meta object.
   *
   * Shows observations, features, groups, and group names.
   *
   * @param out  Output context.
   * @param meta The meta JSON (observations, features, groups).
   */
  void print_data_summary(Output& out, nlohmann::json const& meta);

  /**
   * @brief Display a full model summary from its JSON representation.
   *
   * Shows configuration, data summary, training/OOB confusion matrices,
   * degenerate warnings, timing, and variable importance.
   * Used by both `run_train` (after training) and `run_summarize` (from file).
   */
  void print_summary(Output& out, nlohmann::json const& model_data, ConfigDisplayHints const& hints = {});
}
