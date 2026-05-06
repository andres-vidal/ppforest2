/**
 * @file Presentation.test.cpp
 * @brief Unit tests for ModelStats aggregation and JSON serialization.
 */
#include <gtest/gtest.h>

#include "io/EvaluateResult.hpp"
#include "utils/Types.hpp"

using namespace ppforest2::types;
using ppforest2::io::ModelStats;

// ---------------------------------------------------------------------------
// ModelStats — aggregation and JSON serialization
// ---------------------------------------------------------------------------

/* Mean training time across runs. */
TEST(ModelStats, MeanTime) {
  ModelStats stats;
  stats.tr_times = Vector<long long>(3);
  stats.tr_times << 10, 20, 30;

  EXPECT_FLOAT_EQ(stats.mean_time(), 20.0);
}

/* Mean training error across runs. */
TEST(ModelStats, MeanTrainError) {
  ModelStats stats;
  stats.tr_error = Vector<double>(3);
  stats.tr_error << 0.1, 0.2, 0.3;

  EXPECT_FLOAT_EQ(stats.mean_tr_error(), 0.2);
}

/* Mean test error across runs. */
TEST(ModelStats, MeanTestError) {
  ModelStats stats;
  stats.te_error = Vector<double>(3);
  stats.te_error << 0.05, 0.15, 0.25;

  EXPECT_FLOAT_EQ(stats.mean_te_error(), 0.15);
}

/* Standard deviation of training time is positive for varied inputs. */
TEST(ModelStats, StdTime) {
  ModelStats stats;
  stats.tr_times = Vector<long long>(3);
  stats.tr_times << 10, 20, 30;

  EXPECT_GT(stats.std_time(), 0);
}

/* Standard deviation of training error is positive for varied inputs. */
TEST(ModelStats, StdTrainError) {
  ModelStats stats;
  stats.tr_error = Vector<double>(3);
  stats.tr_error << 0.1, 0.2, 0.3;

  EXPECT_GT(stats.std_tr_error(), 0);
}

/* Standard deviation of test error is positive for varied inputs. */
TEST(ModelStats, StdTestError) {
  ModelStats stats;
  stats.te_error = Vector<double>(3);
  stats.te_error << 0.05, 0.15, 0.25;

  EXPECT_GT(stats.std_te_error(), 0);
}

/* JSON output includes std_time_ms, std_train_error, std_test_error. */
TEST(ModelStats, StdFieldsInJson) {
  ModelStats stats;
  stats.tr_times = Vector<long long>(2);
  stats.tr_times << 10, 20;
  stats.tr_error = Vector<double>(2);
  stats.tr_error << 0.1, 0.3;
  stats.te_error = Vector<double>(2);
  stats.te_error << 0.2, 0.4;

  auto j = stats.to_json();

  EXPECT_TRUE(j.contains("std_time_ms"));
  EXPECT_TRUE(j.contains("std_train_error"));
  EXPECT_TRUE(j.contains("std_test_error"));
  EXPECT_GT(j["std_time_ms"].get<double>(), 0);
  EXPECT_GT(j["std_train_error"].get<double>(), 0);
  EXPECT_GT(j["std_test_error"].get<double>(), 0);
}

/* A single run produces zero standard deviation for all metrics. */
TEST(ModelStats, SingleRunStdZero) {
  ModelStats stats;
  stats.tr_times = Vector<long long>(1);
  stats.tr_times << 10;
  stats.tr_error = Vector<double>(1);
  stats.tr_error << 0.1;
  stats.te_error = Vector<double>(1);
  stats.te_error << 0.2;

  auto j = stats.to_json();

  EXPECT_EQ(j["runs"], 1);
  EXPECT_FLOAT_EQ(j["std_time_ms"].get<float>(), 0.0f);
  EXPECT_FLOAT_EQ(j["std_train_error"].get<float>(), 0.0f);
  EXPECT_FLOAT_EQ(j["std_test_error"].get<float>(), 0.0f);
}

/* Full JSON serialization: means, iterations array, no peak_rss when unset. */
TEST(ModelStats, ToJson) {
  ModelStats stats;
  stats.n        = 150;
  stats.p        = 4;
  stats.g        = 3;
  stats.tr_times = Vector<long long>(2);
  stats.tr_times << 10, 20;
  stats.tr_error = Vector<double>(2);
  stats.tr_error << 0.1, 0.3;
  stats.te_error = Vector<double>(2);
  stats.te_error << 0.2, 0.4;

  auto j = stats.to_json();

  EXPECT_EQ(j["n"], 150);
  EXPECT_EQ(j["p"], 4);
  EXPECT_EQ(j["g"], 3);
  EXPECT_EQ(j["runs"], 2);
  EXPECT_FLOAT_EQ(j["mean_time_ms"].get<float>(), 15.0);
  EXPECT_FLOAT_EQ(j["mean_train_error"].get<float>(), 0.2);
  EXPECT_FLOAT_EQ(j["mean_test_error"].get<float>(), 0.3);
  EXPECT_FALSE(j.contains("peak_rss_bytes"));

  // Per-iteration data
  EXPECT_TRUE(j.contains("iterations"));
  EXPECT_EQ(j["iterations"].size(), 2u);
  EXPECT_FLOAT_EQ(j["iterations"][0]["train_time_ms"].get<float>(), 10.0);
  EXPECT_FLOAT_EQ(j["iterations"][0]["train_error"].get<float>(), 0.1);
  EXPECT_FLOAT_EQ(j["iterations"][0]["test_error"].get<float>(), 0.2);
  EXPECT_FLOAT_EQ(j["iterations"][1]["train_time_ms"].get<float>(), 20.0);
  EXPECT_FLOAT_EQ(j["iterations"][1]["train_error"].get<float>(), 0.3);
  EXPECT_FLOAT_EQ(j["iterations"][1]["test_error"].get<float>(), 0.4);
}

/* JSON includes peak_rss_bytes/mb when set; iterations lack peak_rss. */
TEST(ModelStats, ToJsonWithRSS) {
  ModelStats stats;
  stats.tr_times = Vector<long long>(1);
  stats.tr_times << 100;
  stats.tr_error = Vector<double>(1);
  stats.tr_error << 0.05;
  stats.te_error = Vector<double>(1);
  stats.te_error << 0.1;
  stats.peak_rss_bytes = 10485760L; // 10 MB

  auto j = stats.to_json();

  EXPECT_EQ(j["peak_rss_bytes"], 10485760L);
  EXPECT_NEAR(j["peak_rss_mb"].get<double>(), 10.0, 0.01);

  // Iterations should not contain peak_rss
  EXPECT_EQ(j["iterations"].size(), 1u);
  EXPECT_FALSE(j["iterations"][0].contains("peak_rss"));
}
