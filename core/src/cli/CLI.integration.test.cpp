/**
 * @file CLI.integration.test.cpp
 * @brief Integration tests for global flags, help, config, and end-to-end pipeline.
 */
#include "cli/CLI.integration.hpp"

// ---------------------------------------------------------------------------
// Global flags and help
// ---------------------------------------------------------------------------

/* --help prints subcommand names and exits successfully. */
TEST(CLIGlobal, HelpFlag) {
  auto result = run_ppforest2("--help");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_NE(result.stdout_output.find("train"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("predict"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("evaluate"), std::string::npos);
}

/* --version prints non-empty version string and exits successfully. */
TEST(CLIGlobal, VersionFlag) {
  auto result = run_ppforest2("--version");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_FALSE(result.stdout_output.empty());
}

/* -q with evaluate produces completely empty stdout. */
TEST(CLIGlobal, QuietSuppressesOutput) {
  auto quiet_result = run_ppforest2("-q evaluate --simulate 50x3x2 -n 5 -r 0 -i 1");
  EXPECT_EQ(quiet_result.exit_code, 0);
  EXPECT_TRUE(quiet_result.stdout_output.empty());
}

/* -q suppresses "Evaluation results", "Train Error", "Test Error". */
TEST(CLIGlobal, QuietSuppressesEvaluateResults) {
  auto quiet_result = run_ppforest2("-q evaluate --simulate 50x3x2 -n 5 -r 0 -i 1");
  EXPECT_EQ(quiet_result.exit_code, 0);
  EXPECT_EQ(quiet_result.stdout_output.find("Evaluation results"), std::string::npos);
  EXPECT_EQ(quiet_result.stdout_output.find("Train Error"), std::string::npos);
  EXPECT_EQ(quiet_result.stdout_output.find("Test Error"), std::string::npos);
}

/* No arguments at all exits with a non-zero code. */
TEST(CLIGlobal, NoArgsExitsNonZero) {
  auto result = run_ppforest2("");
  EXPECT_NE(result.exit_code, 0);
}

// ---------------------------------------------------------------------------
// Config File
// ---------------------------------------------------------------------------

/* A JSON config file overrides default parameters. */
TEST(CLIGlobal, ConfigFileApplied) {
  TempFile config;
  {
    std::ofstream out(config.path());
    out << R"({"size": 3})";
  }

  TempFile output;
  output.clear();
  auto result =
      run_ppforest2("--config " + config.path() + " -q evaluate --simulate 50x3x2 -r 0 -i 1 -o " + output.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(output.read());
  EXPECT_TRUE(j.contains("runs"));
}

// ---------------------------------------------------------------------------
// End-to-End Pipeline
// ---------------------------------------------------------------------------

/* Full pipeline: train a forest, then predict on the same data. */
TEST(CLIGlobal, TrainThenPredict) {
  TempFile model;
  model.clear();

  auto train_result = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 -s " + model.path());
  ASSERT_EQ(train_result.exit_code, 0);

  TempFile output;
  output.clear();
  auto predict_result = run_ppforest2("-q predict -M " + model.path() + " -d " + IRIS_CSV + " -o " + output.path());
  ASSERT_EQ(predict_result.exit_code, 0);

  auto j = json::parse(output.read());
  EXPECT_TRUE(j.contains("predictions"));
  EXPECT_GT(j["predictions"].size(), 0u);
  ASSERT_TRUE(j.contains("metrics"));
  EXPECT_TRUE(j["metrics"].contains("error_rate"));
  EXPECT_TRUE(j["metrics"].contains("confusion_matrix"));

  for (auto const& pred : j["predictions"]) {
    EXPECT_TRUE(pred.is_string()) << "predictions should use group name strings";
  }
}
