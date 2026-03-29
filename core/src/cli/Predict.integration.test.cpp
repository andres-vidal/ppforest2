/**
 * @file Predict.integration.test.cpp
 * @brief Integration tests for the predict subcommand.
 */
#include "cli/CLI.integration.hpp"

// ---------------------------------------------------------------------------
// Predict subcommand (fixture-based)
// ---------------------------------------------------------------------------

class PredictTest : public SavedModelTest {};

/* Default predict shows error rate and confusion matrix. */
TEST_F(PredictTest, PredictWithSavedModel) {
  auto result = run_ppforest2("predict -M " + model_->path() + " -d " + IRIS_CSV);
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_FALSE(result.stdout_output.empty());
  EXPECT_NE(result.stdout_output.find("Prediction results for"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("Error:"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("Confusion Matrix:"), std::string::npos);
}

/* --no-metrics in quiet mode suppresses error rate and confusion matrix. */
TEST_F(PredictTest, PredictNoMetrics) {
  auto result = run_ppforest2("-q predict -M " + model_->path() + " -d " + IRIS_CSV + " --no-metrics");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_EQ(result.stdout_output.find("Error:"), std::string::npos);
  EXPECT_EQ(result.stdout_output.find("Confusion Matrix:"), std::string::npos);
}

/* --no-metrics without quiet still suppresses metrics output. */
TEST_F(PredictTest, PredictNoMetricsWithoutQuiet) {
  // --no-metrics without -q should also suppress metrics
  auto result = run_ppforest2("--no-color predict -M " + model_->path() + " -d " + IRIS_CSV + " --no-metrics");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_EQ(result.stdout_output.find("Error:"), std::string::npos);
  EXPECT_EQ(result.stdout_output.find("Confusion Matrix:"), std::string::npos);
}

/* --no-metrics omits error_rate and confusion_matrix from JSON output. */
TEST_F(PredictTest, PredictNoMetricsOutputFile) {
  // --no-metrics should omit error_rate and confusion_matrix from output file
  TempFile output;
  output.clear();
  auto result = run_ppforest2("-q predict -M " + model_->path() + " -d " + IRIS_CSV + " --no-metrics -o " + output.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(output.read());
  EXPECT_TRUE(j.contains("predictions"));
  EXPECT_FALSE(j.contains("error_rate"));
  EXPECT_FALSE(j.contains("confusion_matrix"));
}

/* Without --output, predict shows a hint about --output. */
TEST_F(PredictTest, PredictSuggestsOutputHint) {
  auto result = run_ppforest2("--no-color predict -M " + model_->path() + " -d " + IRIS_CSV);
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_NE(result.stdout_output.find("--output"), std::string::npos);
}

/* When --output is used, the hint is suppressed. */
TEST_F(PredictTest, PredictNoHintWhenOutputUsed) {
  TempFile output;
  output.clear();
  auto result = run_ppforest2("--no-color predict -M " + model_->path() + " -d " + IRIS_CSV + " -o " + output.path());
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_EQ(result.stdout_output.find("Tip:"), std::string::npos);
}

/* Quiet mode produces completely empty stdout. */
TEST_F(PredictTest, PredictQuietSuppressesAll) {
  auto result = run_ppforest2("-q predict -M " + model_->path() + " -d " + IRIS_CSV);
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_TRUE(result.stdout_output.empty());
}

/* -o writes predictions, error_rate, confusion_matrix, and proportions to JSON. */
TEST_F(PredictTest, PredictOutputFile) {
  TempFile output;
  output.clear();
  auto result = run_ppforest2("-q predict -M " + model_->path() + " -d " + IRIS_CSV + " -o " + output.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(output.read());
  EXPECT_TRUE(j.contains("predictions"));
  EXPECT_TRUE(j.contains("error_rate"));
  EXPECT_TRUE(j.contains("confusion_matrix"));
  EXPECT_TRUE(j.contains("proportions"));
}

/* Writing to an existing output file must fail. */
TEST_F(PredictTest, PredictOutputCollisionFails) {
  TempFile output;
  // Don't clear - file exists, should fail
  auto result = run_ppforest2("-q predict -M " + model_->path() + " -d " + IRIS_CSV + " -o " + output.path());
  EXPECT_NE(result.exit_code, 0);
}

/* Forest output includes proportions by default. */
TEST_F(PredictTest, PredictForestIncludesProportions) {
  TempFile output;
  output.clear();
  auto result = run_ppforest2("-q predict -M " + model_->path() + " -d " + IRIS_CSV + " -o " + output.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(output.read());
  EXPECT_TRUE(j.contains("predictions"));
  EXPECT_TRUE(j.contains("proportions"));

  // 150 observations, each row is a vector of proportions
  auto& props = j["proportions"];
  EXPECT_EQ(props.size(), 150u);

  // Each row has one entry per group (iris has 3 groups)
  EXPECT_EQ(props[0].size(), 3u);

  // Each row sums to 1.0
  for (const auto& row : props) {
    double sum = 0;
    for (const auto& val : row) {
      sum += val.get<double>();
    }

    EXPECT_NEAR(sum, 1.0, 1e-6);
  }
}

/* --no-proportions omits proportions from forest output. */
TEST_F(PredictTest, PredictNoProportionsFlag) {
  TempFile output;
  output.clear();
  auto result = run_ppforest2("-q predict -M " + model_->path() + " -d " + IRIS_CSV + " --no-proportions -o " + output.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(output.read());
  EXPECT_TRUE(j.contains("predictions"));
  EXPECT_FALSE(j.contains("proportions"));
}

// ---------------------------------------------------------------------------
// Predict subcommand — standalone tests
// ---------------------------------------------------------------------------

/* Single-tree output includes one-hot proportions. */
TEST(CLIPredict, PredictTreeModelIncludesProportions) {
  TempFile model;
  model.clear();
  auto train = run_ppforest2("-q train -d " + IRIS_CSV + " -t 0 -r 42 -s " + model.path());
  ASSERT_EQ(train.exit_code, 0);

  TempFile output;
  output.clear();
  auto result = run_ppforest2("-q predict -M " + model.path() + " -d " + IRIS_CSV + " -o " + output.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(output.read());
  EXPECT_TRUE(j.contains("predictions"));
  EXPECT_TRUE(j.contains("proportions"));

  // 150 observations, each row is one-hot
  auto& props = j["proportions"];
  EXPECT_EQ(props.size(), 150u);
  EXPECT_EQ(props[0].size(), 3u);

  // Each row sums to 1.0 (one-hot for trees)
  for (const auto& row : props) {
    double sum = 0;
    for (const auto& val : row) {
      sum += val.get<double>();
    }

    EXPECT_NEAR(sum, 1.0, 1e-6);
  }
}

/* --no-proportions omits proportions from tree output. */
TEST(CLIPredict, PredictTreeNoProportionsFlag) {
  TempFile model;
  model.clear();
  auto train = run_ppforest2("-q train -d " + IRIS_CSV + " -t 0 -r 42 -s " + model.path());
  ASSERT_EQ(train.exit_code, 0);

  TempFile output;
  output.clear();
  auto result = run_ppforest2("-q predict -M " + model.path() + " -d " + IRIS_CSV + " --no-proportions -o " + output.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(output.read());
  EXPECT_TRUE(j.contains("predictions"));
  EXPECT_FALSE(j.contains("proportions"));
}

/* confusion_matrix in predict output must not contain "error". */
TEST(CLIPredict, PredictOutputNoErrorInConfusionMatrix) {
  TempFile model;
  model.clear();
  auto train = run_ppforest2("-q train -d " + IRIS_CSV + " -t 5 -r 42 -s " + model.path());
  ASSERT_EQ(train.exit_code, 0);

  TempFile output;
  output.clear();
  auto predict = run_ppforest2("-q predict -M " + model.path() + " -d " + IRIS_CSV + " -o " + output.path());
  ASSERT_EQ(predict.exit_code, 0);

  auto j = json::parse(output.read());
  EXPECT_TRUE(j.contains("confusion_matrix"));
  // confusion_matrix should not have an "error" key (error_rate is at top level)
  EXPECT_FALSE(j["confusion_matrix"].contains("error"));
}

// ---------------------------------------------------------------------------
// Predict subcommand — error cases
// ---------------------------------------------------------------------------

/* Predict without -M must fail. */
TEST(CLIPredict, PredictMissingModelArgFails) {
  auto result = run_ppforest2("predict -d " + IRIS_CSV);
  EXPECT_NE(result.exit_code, 0);
}

/* Predict with a nonexistent model file must fail. */
TEST(CLIPredict, PredictNonexistentModelFails) {
  auto result = run_ppforest2("predict -M /nonexistent.json -d " + IRIS_CSV);
  EXPECT_NE(result.exit_code, 0);
}

/* Predicting with a single-tree model produces correct output. */
TEST(CLIPredict, PredictWithSingleTreeModel) {
  TempFile model;
  model.clear();
  auto train = run_ppforest2("-q train -d " + IRIS_CSV + " -t 0 -r 42 -s " + model.path());
  ASSERT_EQ(train.exit_code, 0);

  TempFile output;
  output.clear();
  auto result = run_ppforest2("-q predict -M " + model.path() + " -d " + IRIS_CSV + " -o " + output.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(output.read());
  EXPECT_TRUE(j.contains("predictions"));
  EXPECT_EQ(j["predictions"].size(), 150u);
  EXPECT_TRUE(j.contains("error_rate"));
  EXPECT_TRUE(j.contains("confusion_matrix"));
}
