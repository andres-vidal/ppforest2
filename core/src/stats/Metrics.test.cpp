#include "stats/Metrics.hpp"

#include "utils/Macros.hpp"

#include <gtest/gtest.h>
#include <stdexcept>

using namespace ppforest2::stats;
using namespace ppforest2::types;

// ---------------------------------------------------------------------------
// Classification metrics
// ---------------------------------------------------------------------------

TEST(Metrics, AccuracyMax) {
  OutcomeVector predictions = VEC(Outcome, 1, 2, 3);
  GroupIdVector actual      = VEC(GroupId, 1, 2, 3);

  float result   = accuracy(predictions, actual);
  float expected = 1.0;

  ASSERT_FLOAT_EQ(expected, result);
}

TEST(Metrics, AccuracyMin) {
  OutcomeVector predictions = VEC(Outcome, 1, 2, 3);
  GroupIdVector actual      = VEC(GroupId, 3, 3, 1);

  float result   = accuracy(predictions, actual);
  float expected = 0.0;

  ASSERT_FLOAT_EQ(expected, result);
}

TEST(Metrics, AccuracyGeneric1) {
  OutcomeVector predictions = VEC(Outcome, 1, 2, 3);
  GroupIdVector actual      = VEC(GroupId, 1, 3, 3);

  float result   = accuracy(predictions, actual);
  float expected = 2.0 / 3.0;

  ASSERT_FLOAT_EQ(expected, result);
}

TEST(Metrics, AccuracyGeneric2) {
  OutcomeVector predictions = VEC(Outcome, 1, 2, 3, 4);
  GroupIdVector actual      = VEC(GroupId, 1, 1, 3, 3);

  float result   = accuracy(predictions, actual);
  float expected = 1.0 / 2.0;

  ASSERT_FLOAT_EQ(expected, result);
}

TEST(Metrics, AccuracyMorePredictionsThanObservations) {
  OutcomeVector predictions  = VEC(Outcome, 0, 1, 2);
  GroupIdVector observations = VEC(GroupId, 0, 1);

  ASSERT_THROW(accuracy(predictions, observations), std::invalid_argument);
}

TEST(Metrics, AccuracyMoreObservationsThanPredictions) {
  OutcomeVector predictions  = VEC(Outcome, 0, 1);
  GroupIdVector observations = VEC(GroupId, 0, 1, 2);

  ASSERT_THROW(accuracy(predictions, observations), std::invalid_argument);
}

TEST(Metrics, ErrorRateMax) {
  OutcomeVector predictions = VEC(Outcome, 1, 2, 3);
  GroupIdVector actual      = VEC(GroupId, 3, 3, 1);

  float result   = error_rate(predictions, actual);
  float expected = 1.0;

  ASSERT_FLOAT_EQ(expected, result);
}

TEST(Metrics, ErrorRateMin) {
  OutcomeVector predictions = VEC(Outcome, 1, 2, 3);
  GroupIdVector actual      = VEC(GroupId, 1, 2, 3);

  float result   = error_rate(predictions, actual);
  float expected = 0.0;

  ASSERT_FLOAT_EQ(expected, result);
}

TEST(Metrics, ErrorRateGeneric1) {
  OutcomeVector predictions = VEC(Outcome, 1, 2, 3);
  GroupIdVector actual      = VEC(GroupId, 1, 3, 3);

  float result   = error_rate(predictions, actual);
  float expected = 1.0 / 3.0;

  ASSERT_FLOAT_EQ(expected, result);
}

TEST(Metrics, ErrorRateGeneric2) {
  OutcomeVector predictions = VEC(Outcome, 1, 2, 3, 4);
  GroupIdVector actual      = VEC(GroupId, 1, 1, 3, 3);

  float result   = error_rate(predictions, actual);
  float expected = 1.0 / 2.0;

  ASSERT_FLOAT_EQ(expected, result);
}

TEST(Metrics, ErrorRateMorePredictionsThanObservations) {
  OutcomeVector predictions  = VEC(Outcome, 0, 1, 2);
  GroupIdVector observations = VEC(GroupId, 0, 1);

  ASSERT_THROW(error_rate(predictions, observations), std::invalid_argument);
}

TEST(Metrics, ErrorRateMoreObservationsThanPredictions) {
  OutcomeVector predictions  = VEC(Outcome, 0, 1);
  GroupIdVector observations = VEC(GroupId, 0, 1, 2);

  ASSERT_THROW(error_rate(predictions, observations), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Regression metrics
// ---------------------------------------------------------------------------

TEST(Metrics, MSE) {
  OutcomeVector predictions = VEC(Outcome, 1.1F, 2.2F, 2.8F, 4.1F, 4.9F);
  OutcomeVector actual      = VEC(Outcome, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F);

  // (0.1² + 0.2² + 0.2² + 0.1² + 0.1²) / 5 = 0.022
  EXPECT_NEAR(mse(predictions, actual), 0.022, 1e-4);
}

TEST(Metrics, MAE) {
  OutcomeVector predictions = VEC(Outcome, 1.1F, 2.2F, 2.8F, 4.1F, 4.9F);
  OutcomeVector actual      = VEC(Outcome, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F);

  // (0.1 + 0.2 + 0.2 + 0.1 + 0.1) / 5 = 0.14
  EXPECT_NEAR(mae(predictions, actual), 0.14, 1e-4);
}

TEST(Metrics, RSquared) {
  OutcomeVector predictions = VEC(Outcome, 1.1F, 2.2F, 2.8F, 4.1F, 4.9F);
  OutcomeVector actual      = VEC(Outcome, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F);

  // SS_res = 0.11, SS_tot = 10.0, R² = 1 - 0.11/10 = 0.989
  EXPECT_NEAR(r_squared(predictions, actual), 0.989, 1e-2);
}

TEST(Metrics, PerfectPredictions) {
  OutcomeVector actual = VEC(Outcome, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F);

  EXPECT_NEAR(mse(actual, actual), 0.0, 1e-10);
  EXPECT_NEAR(mae(actual, actual), 0.0, 1e-10);
  EXPECT_NEAR(r_squared(actual, actual), 1.0, 1e-10);
}

TEST(Metrics, RegressionMetricsStruct) {
  OutcomeVector predictions = VEC(Outcome, 1.1F, 2.2F, 2.8F, 4.1F, 4.9F);
  OutcomeVector actual      = VEC(Outcome, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F);

  RegressionMetrics metrics(predictions, actual);

  EXPECT_NEAR(metrics.mse, 0.022, 1e-4);
  EXPECT_NEAR(metrics.mae, 0.14, 1e-4);
  EXPECT_GT(metrics.r_squared, 0.9);
}

TEST(Metrics, RegressionMetricsDefaultIsZero) {
  RegressionMetrics metrics;

  EXPECT_EQ(metrics.mse, 0.0);
  EXPECT_EQ(metrics.mae, 0.0);
  EXPECT_EQ(metrics.r_squared, 0.0);
}

TEST(Metrics, RegressionThrowsOnSizeMismatch) {
  OutcomeVector predictions = VEC(Outcome, 1.0F, 2.0F, 3.0F);
  OutcomeVector actual      = VEC(Outcome, 1.0F, 2.0F);

  EXPECT_THROW(mse(predictions, actual), std::invalid_argument);
  EXPECT_THROW(mae(predictions, actual), std::invalid_argument);
  EXPECT_THROW(r_squared(predictions, actual), std::invalid_argument);
}

TEST(Metrics, RegressionThrowsOnEmpty) {
  OutcomeVector empty(0);

  EXPECT_THROW(mse(empty, empty), std::invalid_argument);
  EXPECT_THROW(mae(empty, empty), std::invalid_argument);
  EXPECT_THROW(r_squared(empty, empty), std::invalid_argument);
}

TEST(Metrics, RegressionSingleElement) {
  OutcomeVector predictions = VEC(Outcome, 2.0F);
  OutcomeVector actual      = VEC(Outcome, 3.0F);

  EXPECT_NEAR(mse(predictions, actual), 1.0, 1e-6);
  EXPECT_NEAR(mae(predictions, actual), 1.0, 1e-6);
  // SS_tot = 0 with one element → R² falls back to 0.
  EXPECT_NEAR(r_squared(predictions, actual), 0.0, 1e-6);
}

TEST(Metrics, RSquaredConstantActualIsZero) {
  OutcomeVector predictions = VEC(Outcome, 4.0F, 5.0F, 6.0F);
  OutcomeVector actual      = VEC(Outcome, 5.0F, 5.0F, 5.0F);

  // SS_tot = 0 with constant actual → R² falls back to 0.
  EXPECT_NEAR(r_squared(predictions, actual), 0.0, 1e-6);
  EXPECT_NEAR(mse(predictions, actual), 2.0 / 3.0, 1e-6);
}

TEST(Metrics, MSENegativeValues) {
  OutcomeVector predictions = VEC(Outcome, -2.5F, -1.5F, 1.5F);
  OutcomeVector actual      = VEC(Outcome, -3.0F, -1.0F, 2.0F);

  // (0.5² + 0.5² + 0.5²) / 3 = 0.25
  EXPECT_NEAR(mse(predictions, actual), 0.25, 1e-6);
}

// ---------------------------------------------------------------------------
// metrics_from_outcomes — variant factory from in-memory tensors
// ---------------------------------------------------------------------------

TEST(MetricsFromOutcomes, ClassificationProducesClassificationVariant) {
  OutcomeVector predictions = VEC(Outcome, 0, 1, 2, 0);
  OutcomeVector actual      = VEC(Outcome, 0, 1, 1, 0);

  Metrics m = metrics_from_outcomes(predictions, actual, Mode::Classification);

  ASSERT_TRUE(std::holds_alternative<ClassificationMetrics>(m));
  auto const& cm = std::get<ClassificationMetrics>(m);
  EXPECT_NEAR(cm.error_rate(), 0.25, 1e-6);
}

TEST(MetricsFromOutcomes, RegressionProducesRegressionVariant) {
  OutcomeVector predictions = VEC(Outcome, 1.0F, 2.0F, 3.0F);
  OutcomeVector actual      = VEC(Outcome, 1.5F, 2.5F, 3.5F);

  Metrics m = metrics_from_outcomes(predictions, actual, Mode::Regression);

  ASSERT_TRUE(std::holds_alternative<RegressionMetrics>(m));
  auto const& rm = std::get<RegressionMetrics>(m);
  EXPECT_NEAR(rm.mse, 0.25, 1e-6);
}

TEST(MetricsFromOutcomes, ClassificationConfusionMatrixCounts) {
  OutcomeVector predictions = VEC(Outcome, 0, 0, 1, 1, 2);
  OutcomeVector actual      = VEC(Outcome, 0, 1, 1, 2, 2);

  auto m         = metrics_from_outcomes(predictions, actual, Mode::Classification);
  auto const& cm = std::get<ClassificationMetrics>(m).confusion_matrix;

  // Predicted-vs-actual diagonal: (0,0), (1,1), (2,2) — 3 correct out of 5.
  EXPECT_NEAR(cm.error(), 0.4, 1e-6);
}
