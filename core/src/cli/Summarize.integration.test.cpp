/**
 * @file Summarize.integration.test.cpp
 * @brief Integration tests for the summarize subcommand and saved model structure.
 */
#include "cli/CLI.integration.hpp"

#include <fstream>

// ---------------------------------------------------------------------------
// Summarize subcommand
// ---------------------------------------------------------------------------

class SummarizeTest : public SavedModelTest {};

/* Summarize must succeed on a saved model and exit 0. */
TEST_F(SummarizeTest, SummarizeExitCode) {
  auto result = run_ppforest2("-q summarize -M " + model_->path());
  EXPECT_EQ(result.exit_code, 0);
}

/* Summarize with a nonexistent model file must fail. */
TEST(CLISummarize, SummarizeMissingModelFails) {
  auto result = run_ppforest2("-q summarize -M /nonexistent/model.json");
  EXPECT_NE(result.exit_code, 0);
}

// ---------------------------------------------------------------------------
// Saved model JSON structure
// ---------------------------------------------------------------------------

/* Saved classification model's training_metrics must carry the confusion matrix block. */
TEST_F(SummarizeTest, HasTrainingConfusionMatrix) {
  ASSERT_TRUE(model_json_.contains("training_metrics"));
  ASSERT_FALSE(model_json_["training_metrics"].is_null());
  auto const& cm = model_json_["training_metrics"]["confusion_matrix"];
  EXPECT_TRUE(cm.contains("matrix"));
  EXPECT_TRUE(cm.contains("labels"));
  EXPECT_TRUE(cm.contains("group_errors"));
  EXPECT_TRUE(model_json_["training_metrics"].contains("error_rate"));
}

/* Saved classification model's oob_metrics must carry the confusion matrix block. */
TEST_F(SummarizeTest, HasOOBConfusionMatrix) {
  ASSERT_TRUE(model_json_.contains("oob_metrics"));
  ASSERT_FALSE(model_json_["oob_metrics"].is_null());
  auto const& cm = model_json_["oob_metrics"]["confusion_matrix"];
  EXPECT_TRUE(cm.contains("matrix"));
  EXPECT_TRUE(cm.contains("labels"));
  EXPECT_TRUE(cm.contains("group_errors"));
  EXPECT_TRUE(model_json_["oob_metrics"].contains("error_rate"));
}

/* Saved model meta must contain observations and features. */
TEST_F(SummarizeTest, MetaHasDataDimensions) {
  ASSERT_TRUE(model_json_.contains("meta"));
  EXPECT_EQ(model_json_["meta"]["observations"].get<int>(), 150);
  EXPECT_EQ(model_json_["meta"]["features"].get<int>(), 4);
  EXPECT_EQ(model_json_["meta"]["groups"].size(), 3U);
}

/* Saved model must contain training duration. */
TEST_F(SummarizeTest, HasTrainingDuration) {
  EXPECT_TRUE(model_json_.contains("training_duration_ms"));
  EXPECT_GE(model_json_["training_duration_ms"].get<long long>(), 0);
}

/* Saved model must not contain ephemeral fields. */
TEST_F(SummarizeTest, NoEphemeralFields) {
  EXPECT_FALSE(model_json_.contains("save_path"));
}

// ---------------------------------------------------------------------------
// Summarize with --no-metrics model
// ---------------------------------------------------------------------------

/* Summarize succeeds on a model trained with --no-metrics. */
TEST(CLISummarize, SummarizeNoMetricsModel) {
  TempFile const model;
  model.clear();
  auto train = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 --no-metrics -s " + model.path());
  ASSERT_EQ(train.exit_code, 0);

  auto result = run_ppforest2("--no-color summarize -M " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  // Config and data summary should still be shown
  EXPECT_NE(result.stdout_output.find("Random Forest"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("Data Summary"), std::string::npos);

  // Metrics should not be shown
  EXPECT_EQ(result.stdout_output.find("Training Confusion Matrix"), std::string::npos);
  EXPECT_EQ(result.stdout_output.find("OOB"), std::string::npos);
  EXPECT_EQ(result.stdout_output.find("Variable Importance"), std::string::npos);
}

/* Summarize with --data recomputes metrics for a --no-metrics model. */
TEST(CLISummarize, SummarizeWithDataRecomputesMetrics) {
  TempFile const model;
  model.clear();
  auto train = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 --no-metrics -s " + model.path());
  ASSERT_EQ(train.exit_code, 0);

  auto result = run_ppforest2("--no-color summarize -M " + model.path() + " -d " + IRIS_CSV);
  EXPECT_EQ(result.exit_code, 0);

  // Metrics should now be shown
  EXPECT_NE(result.stdout_output.find("Training Confusion Matrix"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("OOB Confusion Matrix"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("Variable Importance"), std::string::npos);
}

/* Summarize --data maps labels through the model's group space: a data file
 * listing classes in a different order than the training file still
 * recomputes metrics. */
TEST(CLISummarize, SummarizeWithReorderedDataRecomputesMetrics) {
  TempFile const train_csv(".csv");
  {
    std::ofstream out(train_csv.path());
    out << "f1,f2,y\n1,1,a\n2,1,a\n3,2,a\n11,5,b\n12,6,b\n13,5,b\n";
  }

  TempFile const model;
  model.clear();
  auto train = run_ppforest2("-q train -d " + train_csv.path() + " -n 5 -r 0 --no-metrics -s " + model.path());
  ASSERT_EQ(train.exit_code, 0);

  // Same rows, class "b" listed first.
  TempFile const data_csv(".csv");
  {
    std::ofstream out(data_csv.path());
    out << "f1,f2,y\n11,5,b\n12,6,b\n13,5,b\n1,1,a\n2,1,a\n3,2,a\n";
  }

  auto result = run_ppforest2("--no-color summarize -M " + model.path() + " -d " + data_csv.path());
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_NE(result.stdout_output.find("Training Confusion Matrix"), std::string::npos);
}

/* Summarize --data with a mismatched feature count fails cleanly. */
TEST(CLISummarize, SummarizeDataFeatureCountMismatchFails) {
  TempFile const model;
  model.clear();
  auto train = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 --no-metrics -s " + model.path());
  ASSERT_EQ(train.exit_code, 0);

  TempFile const data_csv(".csv");
  {
    std::ofstream out(data_csv.path());
    out << "f1,y\n1,a\n2,b\n";
  }

  auto result = run_ppforest2("-q summarize -M " + model.path() + " -d " + data_csv.path());
  EXPECT_NE(result.exit_code, 0);
  EXPECT_NE(result.stderr_output.find("feature column"), std::string::npos);
}

/* Summarize with --data does not recompute when metrics already exist. */
TEST_F(SummarizeTest, SummarizeWithDataSkipsExistingMetrics) {
  // Model already has metrics — providing --data should not change output
  auto result = run_ppforest2("-q summarize -M " + model_->path() + " -d " + IRIS_CSV);
  EXPECT_EQ(result.exit_code, 0);
}

/* Summarize on a regression-saved model renders regression-mode metrics
 * (OOB MSE, Regression Metrics table) without recomputation. The unified
 * `training_metrics` / `oob_metrics` blocks are mode-polymorphic — their
 * inner shape comes from `meta.mode`, not from "which sibling key is
 * present", so the reader / presentation layer never has to guess.
 * Using simulate+--mode regression sidesteps the lack of a regression
 * CSV in `data/` while still exercising the full train → save →
 * summarize pipeline end-to-end. */
TEST(CLISummarize, SummarizeRegressionModelRendersRegressionMetrics) {
  TempFile const config;
  {
    std::ofstream out(config.path());
    // NxPxG format is mandatory, but the G component is ignored by the
    // regression `simulate` overload — the regression path only consumes
    // N and P. Passing `2` keeps the validator happy without changing semantics.
    out << R"({"simulate": "80x3x2", "mode": "regression", "seed": 0, "size": 5})";
  }

  TempFile const model;
  model.clear();
  auto train = run_ppforest2("-q train --config " + config.path() + " -s " + model.path());
  ASSERT_EQ(train.exit_code, 0) << train.stderr_output;

  // Saved JSON's training_metrics carries the regression fields (mse / mae /
  // r_squared); the inner shape is dictated by `config.mode == regression`.
  auto j = json::parse(model.read());
  EXPECT_EQ(j["config"]["mode"], "regression");
  ASSERT_TRUE(j.contains("training_metrics"));
  ASSERT_FALSE(j["training_metrics"].is_null());
  EXPECT_TRUE(j["training_metrics"].contains("mse"));
  EXPECT_FALSE(j["training_metrics"].contains("confusion_matrix"));

  // Summarize without --data: metrics must render from the saved JSON
  // (no recomputation path available — the saved model carries no x/y).
  auto result = run_ppforest2("--no-color summarize -M " + model.path());
  ASSERT_EQ(result.exit_code, 0) << result.stderr_output;
  EXPECT_NE(result.stdout_output.find("Training MSE"), std::string::npos) << "Regression training metrics must render";
  EXPECT_NE(result.stdout_output.find("OOB MSE"), std::string::npos) << "Regression OOB metrics must render";
  EXPECT_EQ(result.stdout_output.find("Confusion Matrix"), std::string::npos)
      << "Regression model must not render confusion-matrix tables";
}

// ---------------------------------------------------------------------------
// Saved model — bootstrap sample indices
// ---------------------------------------------------------------------------

/* Saved forest must contain sample_indices for each tree. */
TEST_F(SummarizeTest, HasSampleIndices) {
  auto trees = model_json_["model"]["trees"];
  ASSERT_GT(trees.size(), 0U);

  for (auto const& tree : trees) {
    ASSERT_TRUE(tree.contains("sample_indices")) << "Each tree must have sample_indices";
    EXPECT_GT(tree["sample_indices"].size(), 0U);
  }
}

// ---------------------------------------------------------------------------
// Summarize with non-default strategies
// ---------------------------------------------------------------------------

/* Summarize shows strategy display names for non-default strategies. */
TEST(CLISummarize, SummarizeNonDefaultStrategies) {
  TempFile const model;
  model.clear();
  auto train = run_ppforest2(
      "-q train -d " + IRIS_CSV +
      " -n 5 -r 0 "
      "--pp pda:lambda=0.5 --vars all -s " +
      model.path()
  );
  ASSERT_EQ(train.exit_code, 0);

  auto result = run_ppforest2("--no-color summarize -M " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  // Strategy display names should appear in summary output
  EXPECT_NE(result.stdout_output.find("PDA"), std::string::npos) << "Expected PDA strategy display name in summary";
}
