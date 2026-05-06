/**
 * @file Evaluate.integration.test.cpp
 * @brief Integration tests for the evaluate subcommand.
 */
#include "cli/CLI.integration.hpp"

// ---------------------------------------------------------------------------
// Evaluate subcommand
// ---------------------------------------------------------------------------

/* Basic evaluation with simulated data succeeds. */
TEST(CLIEvaluate, EvaluateWithSimulatedData) {
  auto result = run_ppforest2("-q evaluate --simulate 50x3x2 -n 5 -r 0 -i 1");
  EXPECT_EQ(result.exit_code, 0);
}

/* Evaluation with real iris data succeeds. */
TEST(CLIEvaluate, EvaluateWithIrisData) {
  auto result = run_ppforest2("-q evaluate -d " + IRIS_CSV + " -n 5 -r 0 -i 1");
  EXPECT_EQ(result.exit_code, 0);
}

/* Non-quiet evaluate prints results header and error metrics. */
TEST(CLIEvaluate, EvaluateTextOutput) {
  auto result = run_ppforest2("evaluate --simulate 50x3x2 -n 5 -r 0 -i 1");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_NE(result.stdout_output.find("Evaluation results"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("Train Err"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("Test Err"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("Peak RSS"), std::string::npos);
}

/* Evaluation with a single tree (t=0) succeeds. */
TEST(CLIEvaluate, EvaluateSingleTree) {
  auto result = run_ppforest2("-q evaluate --simulate 50x3x2 -n 0 -r 0 -i 1");
  EXPECT_EQ(result.exit_code, 0);
}

/* -o writes evaluation stats (runs, mean_time, peak_rss) to JSON. */
TEST(CLIEvaluate, EvaluateOutputFile) {
  TempFile const output;
  output.clear();
  auto result = run_ppforest2("-q evaluate --simulate 50x3x2 -n 5 -r 0 -i 1 -o " + output.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(output.read());
  EXPECT_TRUE(j.contains("runs"));
  EXPECT_TRUE(j.contains("mean_time_ms"));
  EXPECT_TRUE(j.contains("peak_rss_bytes"));
}

/* Multiple iterations must train *distinct* models — otherwise the std-error
 * fields are tautologically zero. Fences the per-iteration seed derivation
 * in `evaluate_model`: each loop iteration uses a fresh seed drawn from the
 * caller's RNG, so bagging variance produces different forests and therefore
 * non-zero error variance across iterations. Iris is the smallest dataset
 * with enough overlap between groups to surface real bagging variance —
 * `--simulate` defaults are too well-separated (forests classify trivially). */
TEST(CLIEvaluate, EvaluateIterationsProduceModelVariance) {
  TempFile const output;
  output.clear();
  auto result = run_ppforest2("-q evaluate -d " + IRIS_CSV + " -n 5 -r 0 -i 5 --train-ratio 0.5 -o " + output.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(output.read());
  EXPECT_GT(j["std_train_error"].get<double>(), 0.0);
  EXPECT_GT(j["std_test_error"].get<double>(), 0.0);
}

/* `--fixed-seed` opts out of per-iteration variance: every iteration trains
 * the same model, so `std_*_error` is exactly zero. Useful for measuring
 * pure timing noise without algorithmic variance. */
TEST(CLIEvaluate, EvaluateFixedSeedProducesZeroErrorVariance) {
  TempFile const output;
  output.clear();
  auto result = run_ppforest2(
      "-q evaluate -d " + IRIS_CSV + " -n 5 -r 0 -i 5 --train-ratio 0.5 --fixed-seed -o " + output.path()
  );
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(output.read());
  EXPECT_DOUBLE_EQ(j["std_train_error"].get<double>(), 0.0);
  EXPECT_DOUBLE_EQ(j["std_test_error"].get<double>(), 0.0);
}

/* -e exports config.json, data.csv, and results.json to a directory. */
TEST(CLIEvaluate, EvaluateExport) {
  TempDir const dir;
  std::string const export_path = dir.path() + "/exp1";

  auto result = run_ppforest2("-q evaluate --simulate 50x3x2 -n 5 -r 0 -l 0.3 -i 1 -e " + export_path);
  EXPECT_EQ(result.exit_code, 0);

  EXPECT_TRUE(std::filesystem::exists(export_path + "/config.json"));
  EXPECT_TRUE(std::filesystem::exists(export_path + "/data.csv"));
  EXPECT_TRUE(std::filesystem::exists(export_path + "/results.json"));

  // Verify config.json captures all configuration for reproducibility
  std::ifstream config_in(export_path + "/config.json");
  auto config = json::parse(config_in);
  EXPECT_EQ(config["data"], "data.csv");
  EXPECT_EQ(config["size"], 5);
  EXPECT_FLOAT_EQ(config["pp"]["lambda"].get<float>(), 0.3F);
  EXPECT_EQ(config["seed"], 0);
  EXPECT_TRUE(config.contains("threads"));
  EXPECT_TRUE(config.contains("train_ratio"));
  EXPECT_TRUE(config.contains("iterations"));
}

/* Output to an existing file must fail. */
TEST(CLIEvaluate, EvaluateOutputCollisionFails) {
  TempFile const output;
  // Don't clear - file exists, should fail
  auto result = run_ppforest2("-q evaluate --simulate 50x3x2 -n 5 -r 0 -i 1 -o " + output.path());
  EXPECT_NE(result.exit_code, 0);
}

/* Export to an existing directory must fail. */
TEST(CLIEvaluate, EvaluateExportCollisionFails) {
  TempDir const dir;
  // dir.path() already exists, should fail
  auto result = run_ppforest2("-q evaluate --simulate 50x3x2 -n 5 -r 0 -i 1 -e " + dir.path());
  EXPECT_NE(result.exit_code, 0);
}

/* JSON output includes an iterations array with per-run metrics. */
TEST(CLIEvaluate, EvaluateOutputHasIterationsArray) {
  TempFile const output;
  output.clear();
  auto result = run_ppforest2("-q evaluate --simulate 50x3x2 -n 5 -r 0 -i 1 -o " + output.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(output.read());
  EXPECT_TRUE(j.contains("iterations"));
  EXPECT_EQ(j["iterations"].size(), 1u);
  EXPECT_TRUE(j["iterations"][0].contains("train_time_ms"));
  EXPECT_TRUE(j["iterations"][0].contains("train_error"));
  EXPECT_TRUE(j["iterations"][0].contains("test_error"));
}

/* Multiple iterations (-i 3) produce matching array size. */
TEST(CLIEvaluate, EvaluateMultipleIterationsArray) {
  TempFile const output;
  output.clear();
  auto result = run_ppforest2("-q evaluate --simulate 50x3x2 -n 5 -r 0 -i 3 -o " + output.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(output.read());
  EXPECT_TRUE(j.contains("iterations"));
  EXPECT_EQ(j["iterations"].size(), 3u);
  EXPECT_EQ(j["runs"], 3);

  for (auto const& iter : j["iterations"]) {
    EXPECT_TRUE(iter.contains("train_time_ms"));
    EXPECT_TRUE(iter.contains("train_error"));
    EXPECT_TRUE(iter.contains("test_error"));
    EXPECT_FALSE(iter.contains("peak_rss"));
  }
}

/* Evaluate without -d or --simulate must fail. */
TEST(CLIEvaluate, EvaluateNoDataSourceFails) {
  auto result = run_ppforest2("evaluate");
  EXPECT_NE(result.exit_code, 0);
}

/* Malformed --simulate string must fail. */
TEST(CLIEvaluate, EvaluateInvalidSimFormatFails) {
  auto result = run_ppforest2("evaluate --simulate 100x5");
  EXPECT_NE(result.exit_code, 0);
}

/* Fraction "1/3" is accepted end-to-end for evaluate. */
TEST(CLIEvaluate, EvaluateWithFractionVars) {
  auto result = run_ppforest2("-q evaluate --simulate 50x3x2 -n 5 -r 0 -i 1 --p-vars 1/3");
  EXPECT_EQ(result.exit_code, 0);
}

// ---------------------------------------------------------------------------
// Explicit strategy flags
// ---------------------------------------------------------------------------

/* Evaluate with --pp and --vars produces correct config in export. */
TEST(CLIEvaluate, EvaluateWithStrategyFlags) {
  TempDir const dir;
  std::string const export_path = dir.path() + "/exp_strategies";

  auto result = run_ppforest2(
      "-q evaluate --simulate 50x3x2 -n 5 -r 0 -i 1 "
      "--pp pda:lambda=0.3 --vars uniform:count=2 --cutpoint mean_of_means "
      "-e " +
      export_path
  );
  EXPECT_EQ(result.exit_code, 0);

  std::ifstream config_in(export_path + "/config.json");
  auto config = json::parse(config_in);
  EXPECT_EQ(config["pp"]["name"], "pda");
  EXPECT_FLOAT_EQ(config["pp"]["lambda"].get<float>(), 0.3F);
  EXPECT_EQ(config["vars"]["name"], "uniform");
  EXPECT_EQ(config["vars"]["count"], 2);
  EXPECT_EQ(config["cutpoint"]["name"], "mean_of_means");
}

/* Evaluate with --vars all succeeds (no variable subsampling). */
TEST(CLIEvaluate, EvaluateWithVarsAll) {
  auto result = run_ppforest2("-q evaluate --simulate 50x3x2 -n 5 -r 0 -i 1 --vars all");
  EXPECT_EQ(result.exit_code, 0);
}

/* Exported config.json can be used as --config for train. */
TEST(CLIEvaluate, ExportedConfigUsableForTraining) {
  TempDir const dir;
  std::string const export_path = dir.path() + "/exp";

  auto eval_result = run_ppforest2(
      "-q evaluate --simulate 50x3x2 -n 5 -r 0 -l 0.3 -i 1 "
      "--vars uniform:count=2 --cutpoint mean_of_means "
      "-e " +
      export_path
  );
  ASSERT_EQ(eval_result.exit_code, 0);

  std::string const config_path = export_path + "/config.json";
  std::string const data_path   = export_path + "/data.csv";

  TempFile model;
  model.clear();
  auto train_result = run_ppforest2("--config " + config_path + " -q train -d " + data_path + " -s " + model.path());
  EXPECT_EQ(train_result.exit_code, 0) << train_result.stderr_output;

  auto j = json::parse(model.read());
  EXPECT_EQ(j["config"]["size"], 5);
  EXPECT_FLOAT_EQ(j["config"]["pp"]["lambda"].get<float>(), 0.3F);
  EXPECT_EQ(j["config"]["vars"]["name"], "uniform");
  EXPECT_EQ(j["config"]["vars"]["count"], 2);
  EXPECT_EQ(j["config"]["cutpoint"]["name"], "mean_of_means");
}
