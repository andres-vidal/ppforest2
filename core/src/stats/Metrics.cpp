#include "stats/Metrics.hpp"

#include <stdexcept>

using namespace ppforest2::types;

namespace ppforest2::stats {
  namespace {
    void validate_inputs(OutcomeVector const& predictions, OutcomeVector const& actual) {
      if (predictions.size() != actual.size()) {
        throw std::invalid_argument("Predictions and actual vectors must have the same size.");
      }

      if (predictions.size() == 0) {
        throw std::invalid_argument("Vectors must not be empty.");
      }
    }
  }

  float accuracy(OutcomeVector const& predictions, GroupIdVector const& actual) {
    if (predictions.rows() != actual.rows()) {
      throw std::invalid_argument("predictions and actual must have the same number of rows");
    }

    auto const matches = (predictions.cast<GroupId>().array() == actual.array()).count();
    return static_cast<float>(matches) / static_cast<float>(predictions.rows());
  }

  double error_rate(OutcomeVector const& predictions, GroupIdVector const& actual) {
    if (predictions.rows() != actual.rows()) {
      throw std::invalid_argument("predictions and actual must have the same number of rows");
    }

    return 1.0 - accuracy(predictions, actual);
  }

  double mse(OutcomeVector const& predictions, OutcomeVector const& actual) {
    validate_inputs(predictions, actual);
    return (predictions - actual).cast<double>().array().square().mean();
  }

  double mae(OutcomeVector const& predictions, OutcomeVector const& actual) {
    validate_inputs(predictions, actual);
    return (predictions - actual).cast<double>().array().abs().mean();
  }

  double r_squared(OutcomeVector const& predictions, OutcomeVector const& actual) {
    validate_inputs(predictions, actual);

    double const mean_actual = static_cast<double>(actual.mean());
    double const ss_res      = (predictions - actual).cast<double>().array().square().sum();
    double const ss_tot      = (actual.cast<double>().array() - mean_actual).square().sum();

    return ss_tot == 0.0 ? 0.0 : 1.0 - ss_res / ss_tot;
  }

  RegressionMetrics::RegressionMetrics(OutcomeVector const& predictions, OutcomeVector const& actual)
      : mse(stats::mse(predictions, actual))
      , mae(stats::mae(predictions, actual))
      , r_squared(stats::r_squared(predictions, actual)) {}

  Metrics metrics_from_outcomes(OutcomeVector const& y_pred, OutcomeVector const& y, Mode mode) {
    if (is_regression(mode)) {
      return RegressionMetrics(y_pred, y);
    } else {
      return ClassificationMetrics(y_pred.cast<GroupId>(), y.cast<GroupId>());
    }
  }
}
