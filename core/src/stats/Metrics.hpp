/**
 * @file Metrics.hpp
 * @brief Mode-specific evaluation metric blocks.
 *
 * Pairs `ClassificationMetrics` (confusion matrix + error rate) with
 * `RegressionMetrics` (MSE / MAE / R²). Both are consumed together by
 * the OOB / variant-based code paths in `models/Evaluation.hpp`,
 * serialization, and presentation, so they live side by side.
 */
#pragma once

#include "stats/ConfusionMatrix.hpp"
#include "utils/Types.hpp"

#include <optional>
#include <variant>


namespace ppforest2::stats {
  /**
   * @brief Classification evaluation metrics.
   *
   * Wraps the confusion matrix and exposes the overall error rate.
   */
  struct ClassificationMetrics {
    using Maybe = std::optional<ClassificationMetrics>;

    ConfusionMatrix confusion_matrix;

    ClassificationMetrics() = default;

    explicit ClassificationMetrics(ConfusionMatrix cm)
        : confusion_matrix(std::move(cm)) {}

    ClassificationMetrics(types::GroupIdVector const& predictions, types::GroupIdVector const& actual)
        : confusion_matrix(predictions, actual) {}

    double error_rate() const { return static_cast<double>(confusion_matrix.error()); }
  };

  /**
   * @brief Regression evaluation metrics.
   *
   * Computes MSE, MAE, and R² from predictions and actual values.
   *
   * @code
   *   RegressionMetrics metrics(predictions, actual);
   *   double mse = metrics.mse;
   *   double r2  = metrics.r_squared;
   * @endcode
   */
  struct RegressionMetrics {
    using Maybe = std::optional<RegressionMetrics>;

    double mse       = 0.0; ///< Mean squared error.
    double mae       = 0.0; ///< Mean absolute error.
    double r_squared = 0.0; ///< Coefficient of determination (R²).

    RegressionMetrics() = default;

    /**
     * @brief Compute metrics from predictions and actual values.
     * @throws std::invalid_argument If sizes differ or vectors are empty.
     */
    RegressionMetrics(types::OutcomeVector const& predictions, types::OutcomeVector const& actual);
  };

  // ---------------------------------------------------------------------------
  // Classification metrics
  // ---------------------------------------------------------------------------

  /**
   * @brief Fraction of predictions matching the ground-truth class label.
   * @return Accuracy in [0, 1].
   */
  float accuracy(types::OutcomeVector const& predictions, types::GroupIdVector const& actual);

  /**
   * @brief Misclassification rate — `1 - accuracy`.
   * @return Error rate in [0, 1].
   */
  double error_rate(types::OutcomeVector const& predictions, types::GroupIdVector const& actual);

  /**
   * @brief Convenience overload: float-typed labels (cast to `GroupId` locally).
   *
   * Used by the unified training pipeline where `y` is carried as `OutcomeVector`
   * for both classification and regression.
   */
  inline double error_rate(types::OutcomeVector const& predictions, types::OutcomeVector const& actual) {
    types::GroupIdVector const actual_int = actual.cast<types::GroupId>();
    return error_rate(predictions, actual_int);
  }

  // ---------------------------------------------------------------------------
  // Regression metrics
  // ---------------------------------------------------------------------------

  /** @brief Mean squared error. */
  double mse(types::OutcomeVector const& predictions, types::OutcomeVector const& actual);

  /** @brief Mean absolute error. */
  double mae(types::OutcomeVector const& predictions, types::OutcomeVector const& actual);

  /** @brief Coefficient of determination (R²). Returns 0 when total variance is 0. */
  double r_squared(types::OutcomeVector const& predictions, types::OutcomeVector const& actual);

  /**
   * @brief Mode-polymorphic metrics block.
   *
   * Carries `ClassificationMetrics` for classification models and
   * `RegressionMetrics` for regression models. Consumers that want the
   * scalar / matrix can `std::visit` the variant instead of branching
   * on which JSON key (or strategy mode) happens to be present.
   */
  using Metrics = std::variant<ClassificationMetrics, RegressionMetrics>;

  /**
   * @brief Build a mode-appropriate `Metrics` variant from in-memory tensors.
   *
   * Sibling to `serialization::metrics_from_json` — same variant, different
   * source. Lets callers compute metrics once and pass the variant to
   * polymorphic consumers (e.g. `print_metrics_block`).
   */
  Metrics metrics_from_outcomes(types::OutcomeVector const& y_pred, types::OutcomeVector const& y, types::Mode mode);
}
