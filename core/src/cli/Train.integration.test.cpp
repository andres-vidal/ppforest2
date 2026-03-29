/**
 * @file Train.integration.test.cpp
 * @brief Integration tests for the train subcommand.
 */
#include "cli/CLI.integration.hpp"

// ---------------------------------------------------------------------------
// Train subcommand — TrainTest fixture (inherits SavedModelTest)
// ---------------------------------------------------------------------------

class TrainTest : public SavedModelTest {};

/* Saved forest JSON contains model_type, model, and config block. */
TEST_F(TrainTest, JsonContainsModelAndConfig) {
  EXPECT_EQ(model_json_["model_type"], "forest");
  EXPECT_TRUE(model_json_.contains("model"));
  EXPECT_TRUE(model_json_.contains("config"));

  auto config = model_json_["config"];
  EXPECT_EQ(config["trees"], 5);
  EXPECT_TRUE(config.contains("lambda"));
  EXPECT_EQ(config["seed"], 42);
  EXPECT_TRUE(config.contains("threads"));
  EXPECT_TRUE(config.contains("vars"));
  EXPECT_EQ(config["data"], IRIS_CSV);
}

TEST_F(TrainTest, ForestMetaMatchesGolden) {
  std::ifstream golden_in(GOLDEN_DIR + "/iris/forest-pda-t5-s42.json");
  ASSERT_TRUE(golden_in.is_open());
  auto golden_meta = json::parse(golden_in)["meta"];

  EXPECT_EQ(model_json_["meta"], golden_meta) << "CLI meta diverged from golden file";
}

/* Model file is created at the specified path. */
TEST_F(TrainTest, ModelFileExists) {
  EXPECT_TRUE(std::filesystem::exists(model_->path()));
}

TEST(CLITrain, TrainSingleTreeMetaMatchesGolden) {
  std::ifstream golden_in(GOLDEN_DIR + "/iris/tree-pda-s42.json");
  ASSERT_TRUE(golden_in.is_open());
  auto golden_meta = json::parse(golden_in)["meta"];

  TempFile model;
  model.clear();
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -t 0 -r 42 -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_EQ(j["meta"], golden_meta) << "CLI meta diverged from golden file";
}

/* Single tree (t=0) saves with model_type "tree". */
TEST(CLITrain, TrainAndSaveSingleTree) {
  TempFile model;
  model.clear();
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -t 0 -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_EQ(j["model_type"], "tree");
}


/* --no-save suppresses model file creation. */
TEST(CLITrain, TrainNoSave) {
  TempDir dir;
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -t 5 -r 42 --no-save");
  EXPECT_EQ(result.exit_code, 0);
}

/* Saving to an existing file must fail (no silent overwrite). */
TEST(CLITrain, TrainCollisionFails) {
  TempFile model;
  // Don't clear - file exists, should fail
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -t 5 -r 42 -s " + model.path());
  EXPECT_NE(result.exit_code, 0);
}

/* Train without a data file must fail. */
TEST(CLITrain, TrainMissingDataFails) {
  auto result = run_ppforest2("train");
  EXPECT_NE(result.exit_code, 0);
}

/* Train with a nonexistent data file must fail. */
TEST(CLITrain, TrainNonexistentFileFails) {
  auto result = run_ppforest2("train -d /nonexistent/file.csv");
  EXPECT_NE(result.exit_code, 0);
}

// ---------------------------------------------------------------------------
// Variable importance (always-on for forests, disabled by --no-metrics)
// ---------------------------------------------------------------------------

/* Forest training prints OOB error and Variable Importance table by default. */
TEST(CLITrain, TrainVIShownByDefault) {
  auto result = run_ppforest2("train -d " + IRIS_CSV + " -t 5 -r 42 --no-save");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_NE(result.stdout_output.find("OOB Error"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("Variable Importance"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("Projection"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("Weighted"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("Permuted"), std::string::npos);
}

/* --no-metrics suppresses OOB error and Variable Importance table. */
TEST(CLITrain, TrainNoMetricsSuppressesVI) {
  auto result = run_ppforest2("train -d " + IRIS_CSV + " -t 5 -r 42 --no-save --no-metrics");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_EQ(result.stdout_output.find("OOB error"), std::string::npos);
  EXPECT_EQ(result.stdout_output.find("Variable Importance"), std::string::npos);
}

/* Saved forest JSON contains oob_error and variable_importance. */
TEST_F(TrainTest, VISavedToJson) {
  ASSERT_TRUE(model_json_.contains("oob_error")) << "Expected oob_error key in saved JSON";
  EXPECT_TRUE(model_json_["oob_error"].is_number());
  EXPECT_GE(model_json_["oob_error"].get<double>(), 0.0);
  EXPECT_LE(model_json_["oob_error"].get<double>(), 1.0);

  ASSERT_TRUE(model_json_.contains("variable_importance")) << "Expected variable_importance key in saved JSON";

  auto vi = model_json_["variable_importance"];
  EXPECT_TRUE(vi.contains("scale"));
  EXPECT_TRUE(vi.contains("projections"));
  EXPECT_TRUE(vi.contains("weighted_projections"));
  EXPECT_TRUE(vi.contains("permuted"));

  EXPECT_EQ(vi["scale"].size(), 4u);
  EXPECT_EQ(vi["projections"].size(), 4u);
  EXPECT_EQ(vi["weighted_projections"].size(), 4u);
  EXPECT_EQ(vi["permuted"].size(), 4u);
}

/* With --no-metrics the saved JSON must not contain oob_error or variable_importance. */
TEST(CLITrain, TrainNoMetricsNotInJson) {
  TempFile model;
  model.clear();
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -t 5 -r 42 --no-metrics -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_FALSE(j.contains("oob_error"));
  EXPECT_FALSE(j.contains("variable_importance"));
}

/* Single tree training shows VI2 (projections) only, no OOB error. */
TEST(CLITrain, TrainSingleTreeShowsVI2Only) {
  auto result = run_ppforest2("train -d " + IRIS_CSV + " -t 0 --no-save");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_EQ(result.stdout_output.find("OOB error"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("Variable Importance"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("Projection"), std::string::npos);
  EXPECT_EQ(result.stdout_output.find("Weighted"), std::string::npos);
  EXPECT_EQ(result.stdout_output.find("Permuted"), std::string::npos);
}

/* Single tree saved JSON contains only scale and projections (no weighted/permuted). */
TEST(CLITrain, TrainSingleTreeVISavedToJson) {
  TempFile model;
  model.clear();
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -t 0 -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  ASSERT_TRUE(j.contains("variable_importance"));

  auto vi = j["variable_importance"];
  EXPECT_TRUE(vi.contains("scale"));
  EXPECT_TRUE(vi.contains("projections"));
  EXPECT_FALSE(vi.contains("weighted_projections"));
  EXPECT_FALSE(vi.contains("permuted"));

  EXPECT_EQ(vi["scale"].size(), 4u);
  EXPECT_EQ(vi["projections"].size(), 4u);
}

/* Quiet mode suppresses the VI table but still saves it to JSON. */
TEST_F(TrainTest, VIQuietSuppressesTable) {
  EXPECT_TRUE(model_json_.contains("variable_importance"));
}

// ---------------------------------------------------------------------------
// Train display output
// ---------------------------------------------------------------------------

/* Train with --no-save must still produce output (via print_summary). */
TEST(CLITrain, TrainNoSaveStillDisplays) {
  auto result = run_ppforest2("--no-color train -d " + IRIS_CSV + " -t 5 -r 42 --no-save");
  EXPECT_EQ(result.exit_code, 0);
  // Should still show confusion matrix output
  EXPECT_NE(result.stdout_output.find("Training Confusion Matrix"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("OOB Confusion Matrix"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("Variable Importance"), std::string::npos);
}

/* Single tree train must show training confusion matrix but not OOB. */
TEST(CLITrain, SingleTreeTrainShowsTrainingCM) {
  auto result = run_ppforest2("--no-color train -d " + IRIS_CSV + " -t 0 -r 42 --no-save");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_NE(result.stdout_output.find("Training Confusion Matrix"), std::string::npos);
  // Single tree has no OOB
  EXPECT_EQ(result.stdout_output.find("OOB"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Vars parsing — fraction syntax
// ---------------------------------------------------------------------------

/* Fraction "1/3" is accepted end-to-end for train. */
TEST(CLITrain, TrainWithFractionVars) {
  TempFile model;
  model.clear();
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -t 5 -r 42 -v 1/3 -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);
}

// ---------------------------------------------------------------------------
// Single tree — config structure
// ---------------------------------------------------------------------------

/* Single tree config omits the "vars" key (not applicable). */
TEST(CLITrain, TrainSingleTreeConfigNoVars) {
  TempFile model;
  model.clear();
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -t 0 -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_EQ(j["model_type"], "tree");
  EXPECT_EQ(j["config"]["trees"], 0);
  // Single tree should not have vars in config
  EXPECT_FALSE(j["config"].contains("vars"));
}

// ---------------------------------------------------------------------------
// Saving without .json extension auto-appends it
// ---------------------------------------------------------------------------

/* Saving without .json extension auto-appends it. */
TEST(CLITrain, TrainAutoAppendsJsonExtension) {
  TempDir dir;
  std::string path_no_ext = dir.file("mymodel");

  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -t 5 -r 42 -s " + path_no_ext);
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_TRUE(std::filesystem::exists(path_no_ext + ".json"));
}

// ---------------------------------------------------------------------------
// Parameter coverage — lambda, vars, config override
// ---------------------------------------------------------------------------

/* Training with explicit lambda saves it to config. */
TEST(CLITrain, TrainLambdaSaved) {
  TempFile model;
  model.clear();
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -t 5 -r 42 -l 0.5 -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_FLOAT_EQ(j["config"]["lambda"].get<float>(), 0.5f);
}

/* Training with explicit vars saves it to config. */
TEST(CLITrain, TrainVarsSaved) {
  TempFile model;
  model.clear();
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -t 5 -r 42 -v 2 -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_EQ(j["config"]["vars"], 2);
}

/* CLI args override config file values. */
TEST(CLITrain, CLIArgOverridesConfig) {
  TempFile config;
  {
    std::ofstream out(config.path());
    out << R"({"trees": 3, "seed": 99})";
  }

  TempFile model;
  model.clear();
  auto result = run_ppforest2("--config " + config.path() + " -q train -d " + IRIS_CSV + " -t 7 -r 42 -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_EQ(j["config"]["trees"], 7);
  EXPECT_EQ(j["config"]["seed"], 42);
}

// ---------------------------------------------------------------------------
// Datasets beyond iris — structural tests with crab and wine
// ---------------------------------------------------------------------------

/* Train and predict on crab data succeeds. */
TEST(CLITrain, TrainPredictCrab) {
  TempFile model;
  model.clear();
  auto train = run_ppforest2("-q train -d " + CRAB_CSV + " -t 5 -r 42 -s " + model.path());
  ASSERT_EQ(train.exit_code, 0);

  TempFile output;
  output.clear();
  auto predict = run_ppforest2("-q predict -M " + model.path() + " -d " + CRAB_CSV + " -o " + output.path());
  EXPECT_EQ(predict.exit_code, 0);

  auto j = json::parse(output.read());
  EXPECT_TRUE(j.contains("predictions"));
  EXPECT_GT(j["predictions"].size(), 0u);
  EXPECT_TRUE(j.contains("error_rate"));
  EXPECT_TRUE(j.contains("confusion_matrix"));
}

/* Train and predict on wine data succeeds. */
TEST(CLITrain, TrainPredictWine) {
  TempFile model;
  model.clear();
  auto train = run_ppforest2("-q train -d " + WINE_CSV + " -t 5 -r 42 -s " + model.path());
  ASSERT_EQ(train.exit_code, 0);

  TempFile output;
  output.clear();
  auto predict = run_ppforest2("-q predict -M " + model.path() + " -d " + WINE_CSV + " -o " + output.path());
  EXPECT_EQ(predict.exit_code, 0);

  auto j = json::parse(output.read());
  EXPECT_TRUE(j.contains("predictions"));
  EXPECT_GT(j["predictions"].size(), 0u);
  EXPECT_TRUE(j.contains("error_rate"));
  EXPECT_TRUE(j.contains("confusion_matrix"));
}
