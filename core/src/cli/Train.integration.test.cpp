/**
 * @file Train.integration.test.cpp
 * @brief Integration tests for the train subcommand.
 */
#include "cli/CLI.integration.hpp"

#include "utils/Macros.hpp"

#include <fstream>

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
  EXPECT_EQ(config["size"], 5);
  EXPECT_TRUE(config.contains("pp"));
  EXPECT_TRUE(config["pp"].contains("lambda"));
  EXPECT_EQ(config["seed"], 0);
  EXPECT_TRUE(config.contains("threads"));
  EXPECT_TRUE(config.contains("vars"));
  EXPECT_EQ(config["data"], IRIS_CSV);
}

TEST_F(TrainTest, ForestMetaMatchesGolden) {
  std::ifstream golden_in(GOLDEN_DIR + "/iris/forest-pda-n5-s0.json");
  ASSERT_TRUE(golden_in.is_open());
  auto golden_meta = json::parse(golden_in)["meta"];

  EXPECT_EQ(model_json_["meta"], golden_meta) << "CLI meta diverged from golden file";
}

/* Model file is created at the specified path. */
TEST_F(TrainTest, ModelFileExists) {
  EXPECT_TRUE(std::filesystem::exists(model_->path()));
}

TEST(CLITrain, TrainSingleTreeMetaMatchesGolden) {
  std::ifstream golden_in(GOLDEN_DIR + "/iris/tree-pda-s0.json");
  ASSERT_TRUE(golden_in.is_open());
  auto golden_meta = json::parse(golden_in)["meta"];

  TempFile const model;
  model.clear();
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -n 0 -r 0 -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_EQ(j["meta"], golden_meta) << "CLI meta diverged from golden file";
}

/* Single tree (t=0) saves with model_type "tree". */
TEST(CLITrain, TrainAndSaveSingleTree) {
  TempFile const model;
  model.clear();
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -n 0 -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_EQ(j["model_type"], "tree");
}


/* --no-save suppresses model file creation. */
TEST(CLITrain, TrainNoSave) {
  TempDir const dir;
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 --no-save");
  EXPECT_EQ(result.exit_code, 0);
}

/* Saving to an existing file must fail (no silent overwrite). */
TEST(CLITrain, TrainCollisionFails) {
  TempFile const model;
  // Don't clear - file exists, should fail
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 -s " + model.path());
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

/* Simulated classification saves to JSON and round-trips via summarize
 * without hitting UB. Regression test for a bug where simulated
 * classification produced an empty `group_names` vector, which cascaded
 * into out-of-bounds reads in the labeled leaf serializer. */
TEST(CLITrain, TrainSimulatedClassificationRoundTrip) {
  TempFile const config;
  {
    std::ofstream out(config.path());
    out << R"({"simulate": "50x3x2", "seed": 0, "size": 5})";
  }

  TempFile const model;
  model.clear();

  auto result = run_ppforest2("-q train --config " + config.path() + " -s " + model.path());
  ASSERT_EQ(result.exit_code, 0) << result.stderr_output;

  auto j = json::parse(model.read());
  EXPECT_EQ(j["config"]["mode"], "classification");

  // Simulated classification must produce non-empty `meta.groups` (the
  // writer synthesizes placeholder names, falling back to the integer
  // index as a string). The reader invariant rejects empty `meta.groups`
  // for classification models, so this doubles as a round-trip probe.
  ASSERT_TRUE(j["meta"].contains("groups"));
  EXPECT_EQ(j["meta"]["groups"].size(), 2u);

  // Summarize loads the saved model — if the JSON is malformed or the
  // reader invariant fires, this fails.
  auto summarize = run_ppforest2("-q summarize --model " + model.path());
  EXPECT_EQ(summarize.exit_code, 0) << summarize.stderr_output;
}

// ---------------------------------------------------------------------------
// Variable importance (always-on for forests, disabled by --no-metrics)
// ---------------------------------------------------------------------------

/* Forest training prints OOB error and Variable Importance table by default. */
TEST(CLITrain, TrainVIShownByDefault) {
  auto result = run_ppforest2("train -d " + IRIS_CSV + " -n 5 -r 0 --no-save");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_NE(result.stdout_output.find("OOB Error"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("Variable Importance"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("Projection"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("Weighted"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("Permuted"), std::string::npos);
}

/* --no-metrics suppresses OOB error and Variable Importance table. */
TEST(CLITrain, TrainNoMetricsSuppressesVI) {
  auto result = run_ppforest2("train -d " + IRIS_CSV + " -n 5 -r 0 --no-save --no-metrics");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_EQ(result.stdout_output.find("OOB error"), std::string::npos);
  EXPECT_EQ(result.stdout_output.find("Variable Importance"), std::string::npos);
}

/* Saved forest JSON contains oob_metrics (with classification error_rate) and variable_importance. */
TEST_F(TrainTest, VISavedToJson) {
  ASSERT_TRUE(model_json_.contains("oob_metrics")) << "Expected oob_metrics key in saved JSON";
  ASSERT_FALSE(model_json_["oob_metrics"].is_null());
  ASSERT_TRUE(model_json_["oob_metrics"].contains("error_rate"));
  double const err = model_json_["oob_metrics"]["error_rate"].get<double>();
  EXPECT_GE(err, 0.0);
  EXPECT_LE(err, 1.0);

  ASSERT_TRUE(model_json_.contains("variable_importance")) << "Expected variable_importance key in saved JSON";

  auto vi = model_json_["variable_importance"];
  EXPECT_TRUE(vi.contains("scale"));
  EXPECT_TRUE(vi.contains("projections"));
  EXPECT_TRUE(vi.contains("weighted_projections"));
  EXPECT_TRUE(vi.contains("permuted"));

  EXPECT_EQ(vi["scale"].size(), 4U);
  EXPECT_EQ(vi["projections"].size(), 4U);
  EXPECT_EQ(vi["weighted_projections"].size(), 4U);
  EXPECT_EQ(vi["permuted"].size(), 4U);
}

/* With --no-metrics the saved JSON must expose oob_metrics / variable_importance
 * as `null` (the canonical nullopt-as-null convention). The keys are always
 * present after a fresh save so downstream tooling can distinguish "field
 * understood but no value" from "field missing because of a writer bug". */
TEST(CLITrain, TrainNoMetricsEmitsNullInJson) {
  TempFile const model;
  model.clear();
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 --no-metrics -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  ASSERT_TRUE(j.contains("oob_metrics"));
  EXPECT_TRUE(j["oob_metrics"].is_null());
  ASSERT_TRUE(j.contains("variable_importance"));
  EXPECT_TRUE(j["variable_importance"].is_null());
}

/* Single tree training shows VI2 (projections) only, no OOB error. */
TEST(CLITrain, TrainSingleTreeShowsVI2Only) {
  auto result = run_ppforest2("train -d " + IRIS_CSV + " -n 0 --no-save");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_EQ(result.stdout_output.find("OOB error"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("Variable Importance"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("Projection"), std::string::npos);
  EXPECT_EQ(result.stdout_output.find("Weighted"), std::string::npos);
  EXPECT_EQ(result.stdout_output.find("Permuted"), std::string::npos);
}

/* Single tree saved JSON contains only scale and projections (no weighted/permuted). */
TEST(CLITrain, TrainSingleTreeVISavedToJson) {
  TempFile const model;
  model.clear();
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -n 0 -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  ASSERT_TRUE(j.contains("variable_importance"));

  auto vi = j["variable_importance"];
  EXPECT_TRUE(vi.contains("scale"));
  EXPECT_TRUE(vi.contains("projections"));
  EXPECT_FALSE(vi.contains("weighted_projections"));
  EXPECT_FALSE(vi.contains("permuted"));

  EXPECT_EQ(vi["scale"].size(), 4U);
  EXPECT_EQ(vi["projections"].size(), 4U);
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
  auto result = run_ppforest2("--no-color train -d " + IRIS_CSV + " -n 5 -r 0 --no-save");
  EXPECT_EQ(result.exit_code, 0);
  // Should still show confusion matrix output
  EXPECT_NE(result.stdout_output.find("Training Confusion Matrix"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("OOB Confusion Matrix"), std::string::npos);
  EXPECT_NE(result.stdout_output.find("Variable Importance"), std::string::npos);
}

/* Single tree train must show training confusion matrix but not OOB. */
TEST(CLITrain, SingleTreeTrainShowsTrainingCM) {
  auto result = run_ppforest2("--no-color train -d " + IRIS_CSV + " -n 0 -r 0 --no-save");
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
  TempFile const model;
  model.clear();
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 --p-vars 1/3 -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);
}

// ---------------------------------------------------------------------------
// Single tree — config structure
// ---------------------------------------------------------------------------

/* Single tree config omits the "vars" key (not applicable). */
TEST(CLITrain, TrainSingleTreeConfigNoVars) {
  TempFile const model;
  model.clear();
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -n 0 -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_EQ(j["model_type"], "tree");
  EXPECT_EQ(j["config"]["size"], 0);
  // Single tree uses all vars strategy
  EXPECT_EQ(j["config"]["vars"]["name"], "all");
}

// ---------------------------------------------------------------------------
// Saving without .json extension auto-appends it
// ---------------------------------------------------------------------------

/* Saving without .json extension auto-appends it. */
TEST(CLITrain, TrainAutoAppendsJsonExtension) {
  TempDir const dir;
  std::string const path_no_ext = dir.file("mymodel");

  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 -s " + path_no_ext);
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_TRUE(std::filesystem::exists(path_no_ext + ".json"));
}

// ---------------------------------------------------------------------------
// Parameter coverage — lambda, vars, config override
// ---------------------------------------------------------------------------

/* Training with explicit lambda saves it to config. */
TEST(CLITrain, TrainLambdaSaved) {
  TempFile const model;
  model.clear();
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 -l 0.5 -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_FLOAT_EQ(j["config"]["pp"]["lambda"].get<float>(), 0.5F);
}

/* Training with explicit vars saves it to config. */
TEST(CLITrain, TrainVarsSaved) {
  TempFile const model;
  model.clear();
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 --n-vars 2 -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_EQ(j["config"]["vars"]["count"], 2);
}

/* CLI args override config file values. */
TEST(CLITrain, CLIArgOverridesConfig) {
  TempFile const config;
  {
    std::ofstream out(config.path());
    out << R"({"size": 3, "seed": 99})";
  }

  TempFile const model;
  model.clear();
  auto result =
      run_ppforest2("--config " + config.path() + " -q train -d " + IRIS_CSV + " -n 7 -r 0 -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_EQ(j["config"]["size"], 7);
  EXPECT_EQ(j["config"]["seed"], 0);
}

// ---------------------------------------------------------------------------
// Config file — explicit strategy format
// ---------------------------------------------------------------------------

/* Config file accepts structured strategy format (pp/dr/sr objects). */
TEST(CLITrain, ConfigFileExplicitStrategies) {
  TempFile const config;
  {
    std::ofstream out(config.path());
    out << R"({
      "pp": { "name": "pda", "lambda": 0.3 },
      "vars": { "name": "uniform", "count": 2 },
      "cutpoint": { "name": "mean_of_means" },
      "size": 5,
      "seed": 0
    })";
  }

  TempFile const model;
  model.clear();
  auto result = run_ppforest2("--config " + config.path() + " -q train -d " + IRIS_CSV + " -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_EQ(j["config"]["size"], 5);
  EXPECT_FLOAT_EQ(j["config"]["pp"]["lambda"].get<float>(), 0.3F);
  EXPECT_EQ(j["config"]["vars"]["count"], 2);
}

/* Config file accepts threads key. */
TEST(CLITrain, ConfigFileThreadsKey) {
  TempFile const config;
  {
    std::ofstream out(config.path());
    out << R"({"size": 5, "seed": 0, "threads": 1})";
  }

  TempFile const model;
  model.clear();
  auto result = run_ppforest2("--config " + config.path() + " -q train -d " + IRIS_CSV + " -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_EQ(j["config"]["threads"], 1);
}

// ---------------------------------------------------------------------------
// Datasets beyond iris — structural tests with crab and wine
// ---------------------------------------------------------------------------

/* Train and predict on crab data succeeds. */
TEST(CLITrain, TrainPredictCrab) {
  TempFile const model;
  model.clear();
  auto train = run_ppforest2("-q train -d " + CRAB_CSV + " -n 5 -r 0 -s " + model.path());
  ASSERT_EQ(train.exit_code, 0);

  TempFile const output;
  output.clear();
  auto predict = run_ppforest2("-q predict -M " + model.path() + " -d " + CRAB_CSV + " -o " + output.path());
  EXPECT_EQ(predict.exit_code, 0);

  auto j = json::parse(output.read());
  EXPECT_TRUE(j.contains("predictions"));
  EXPECT_GT(j["predictions"].size(), 0U);
  ASSERT_TRUE(j.contains("metrics"));
  EXPECT_TRUE(j["metrics"].contains("error_rate"));
  EXPECT_TRUE(j["metrics"].contains("confusion_matrix"));
}

/* Regression train → predict round-trip on a real dataset (mtcars). Fences
 * the end-to-end CLI path for `--mode regression` with a small real CSV:
 * training reads the last column (`mpg`) as the continuous response, the
 * saved JSON carries regression-shaped metrics, and `predict` outputs
 * numeric predictions plus MSE/MAE/R² rather than a confusion matrix.
 * Complements the simulated-data tests (which exercise the core pipeline)
 * by catching bugs that depend on real-world feature distributions,
 * non-synthetic column names, or the CSV reader's numeric-response path. */
TEST(CLITrain, TrainRegressionIntegerResponseKeepsNumericY) {
  // Every response value is integer-written, which the reader's written-form
  // detection would classify. --mode regression must override the detection
  // so the model trains on the numeric response, not on label codes.
  TempFile const data(".csv");
  {
    std::ofstream out(data.path());
    out << "f1,f2,y\n1,1,10\n2,2,20\n3,3,30\n4,4,40\n5,5,50\n6,6,60\n";
  }

  TempFile const model;
  model.clear();
  auto train = run_ppforest2("-q train --mode regression -d " + data.path() + " -n 0 -r 0 -s " + model.path());
  ASSERT_EQ(train.exit_code, 0) << train.stderr_output;

  auto mj = json::parse(model.read());
  EXPECT_EQ(mj["config"]["mode"], "regression");
  ASSERT_TRUE(mj["meta"].contains("groups"));
  EXPECT_TRUE(mj["meta"]["groups"].empty());
}

TEST(CLITrain, TrainPredictRegressionMtcars) {
  TempFile const model;
  model.clear();
  auto train = run_ppforest2("-q train --mode regression -d " + MTCARS_CSV + " -n 5 -r 0 -s " + model.path());
  ASSERT_EQ(train.exit_code, 0) << train.stderr_output;

  // Saved JSON's training_metrics block must carry regression fields
  // (mse / mae / r_squared) — inner shape is selected by the model's
  // `config.mode` at deserialization time.
  auto mj = json::parse(model.read());
  EXPECT_EQ(mj["config"]["mode"], "regression");

  // Regression has no group concept — `meta.groups` is emitted as an
  // empty array (matching the in-memory `Names{}`), not as `null`.
  ASSERT_TRUE(mj["meta"].contains("groups"));
  EXPECT_TRUE(mj["meta"]["groups"].is_array());
  EXPECT_TRUE(mj["meta"]["groups"].empty());

  ASSERT_TRUE(mj.contains("training_metrics"));
  ASSERT_FALSE(mj["training_metrics"].is_null());
  EXPECT_TRUE(mj["training_metrics"].contains("mse"));
  EXPECT_TRUE(mj["training_metrics"].contains("mae"));
  EXPECT_TRUE(mj["training_metrics"].contains("r_squared"));
  EXPECT_FALSE(mj["training_metrics"].contains("confusion_matrix"));

  TempFile const output;
  output.clear();
  auto predict = run_ppforest2("-q predict -M " + model.path() + " -d " + MTCARS_CSV + " -o " + output.path());
  ASSERT_EQ(predict.exit_code, 0) << predict.stderr_output;

  auto pj = json::parse(output.read());
  EXPECT_TRUE(pj.contains("predictions"));
  EXPECT_EQ(pj["predictions"].size(), 32U); // mtcars has 32 rows
  // Regression predictions are numeric floats, never group-id ints.
  EXPECT_TRUE(pj["predictions"][0].is_number());
  // Predict JSON uses a single `metrics` key with mode-polymorphic shape:
  // regression carries mse/mae/r_squared; classification would carry
  // error_rate + confusion_matrix instead.
  ASSERT_TRUE(pj.contains("metrics"));
  EXPECT_TRUE(pj["metrics"].contains("mse"));
  EXPECT_TRUE(pj["metrics"].contains("mae"));
  EXPECT_TRUE(pj["metrics"].contains("r_squared"));
  EXPECT_GE(pj["metrics"]["mse"].get<double>(), 0.0);
  EXPECT_FALSE(pj["metrics"].contains("confusion_matrix"));
  EXPECT_FALSE(pj["metrics"].contains("error_rate"));
}

/* Repeated `--stop` on the CLI composes into an `any(...)` CompositeStop
 * end-to-end: train → save → reload → inspect JSON. Fences the main
 * ergonomic fix — users configuring the regression default's composite
 * stop without dropping to `--config`. */
TEST(CLITrain, TrainStopRepeatedComposesAny) {
  TempFile const model;
  model.clear();
  auto train = run_ppforest2(
      "-q train --mode regression -d " + MTCARS_CSV +
      " -n 5 -r 0 "
      "--stop min_size:min_size=5 "
      "--stop min_variance:threshold=0.01 "
      "-s " +
      model.path()
  );
  ASSERT_EQ(train.exit_code, 0) << train.stderr_output;

  auto j = json::parse(model.read());
  ASSERT_TRUE(j["config"].contains("stop"));
  EXPECT_EQ(j["config"]["stop"]["name"], "any");
  ASSERT_EQ(j["config"]["stop"]["rules"].size(), 2U);
  EXPECT_EQ(j["config"]["stop"]["rules"][0]["name"], "min_size");
  EXPECT_EQ(j["config"]["stop"]["rules"][1]["name"], "min_variance");
}

/* CLI `--stop` fully overrides a JSON config's `stop` — including nested
 * `rules[]` when the config held a CompositeStop. No merging happens; the
 * CLI-sourced value wins atomically. This nails the precedence contract
 * between `--config` and `--stop` (same as every other strategy flag but
 * worth fencing explicitly because `stop` is the one with a nested
 * shape). If someone ever changes the semantics to *append* instead of
 * replace (see the landmine comment in ModelParams.cpp::constructor),
 * this test will fire. */
TEST(CLITrain, TrainStopCLICompositeOverridesConfigComposite) {
  TempFile const config;
  {
    std::ofstream out(config.path());
    // Regression dataset: `min_variance` in the CLI override below is a
    // regression-only strategy, so the whole scenario has to run under
    // `--mode regression`. JSON config carries a composite with a single
    // `min_size` rule to prove the CLI replaces it entirely (nested rules
    // and all), not merges into it.
    out << R"({
      "data": ")"
        << MTCARS_CSV << R"(",
      "mode": "regression",
      "size": 5,
      "seed": 0,
      "stop": {
        "name": "any",
        "rules": [ { "name": "min_size", "min_size": 5 } ]
      }
    })";
  }
  TempFile const model;
  model.clear();
  auto train = run_ppforest2(
      "-q train --config " + config.path() + " --stop min_size:min_size=20 --stop min_variance:threshold=0.5 -s " +
      model.path()
  );
  ASSERT_EQ(train.exit_code, 0) << train.stderr_output;

  auto j = json::parse(model.read());
  ASSERT_TRUE(j["config"].contains("stop"));
  auto const& stop = j["config"]["stop"];
  EXPECT_EQ(stop["name"], "any");
  // Must be CLI's two rules, not the config's one rule.
  ASSERT_EQ(stop["rules"].size(), 2U) << "CLI composite should fully replace config composite";
  EXPECT_EQ(stop["rules"][0]["name"], "min_size");
  EXPECT_EQ(stop["rules"][0]["min_size"].get<int>(), 20); // CLI value, not config's 5
  EXPECT_EQ(stop["rules"][1]["name"], "min_variance");
}

/* Single `--stop` keeps the flat non-wrapped shape in the saved model —
 * guards against accidentally wrapping every single-rule config in an
 * any(...) with one rule (which would still work but would pollute
 * serialized JSON with a redundant wrapper). */
TEST(CLITrain, TrainStopSingleNoWrap) {
  TempFile const model;
  model.clear();
  auto train = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 --stop pure_node -s " + model.path());
  ASSERT_EQ(train.exit_code, 0) << train.stderr_output;

  auto j = json::parse(model.read());
  EXPECT_EQ(j["config"]["stop"]["name"], "pure_node");
  EXPECT_FALSE(j["config"]["stop"].contains("rules"));
}

/* Train and predict on wine data succeeds. */
TEST(CLITrain, TrainPredictWine) {
  TempFile const model;
  model.clear();
  auto train = run_ppforest2("-q train -d " + WINE_CSV + " -n 5 -r 0 -s " + model.path());
  ASSERT_EQ(train.exit_code, 0);

  TempFile const output;
  output.clear();
  auto predict = run_ppforest2("-q predict -M " + model.path() + " -d " + WINE_CSV + " -o " + output.path());
  EXPECT_EQ(predict.exit_code, 0);

  auto j = json::parse(output.read());
  EXPECT_TRUE(j.contains("predictions"));
  EXPECT_GT(j["predictions"].size(), 0U);
  ASSERT_TRUE(j.contains("metrics"));
  EXPECT_TRUE(j["metrics"].contains("error_rate"));
  EXPECT_TRUE(j["metrics"].contains("confusion_matrix"));
}

// ---------------------------------------------------------------------------
// Explicit strategy flags (--pp, --vars, --cutpoint)
// ---------------------------------------------------------------------------

/* Train with --pp pda:lambda=0.3 saves PDA lambda in config. */
TEST(CLITrain, TrainWithPPStrategy) {
  TempFile const model;
  model.clear();
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 --pp pda:lambda=0.3 -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_EQ(j["config"]["pp"]["name"], "pda");
  EXPECT_FLOAT_EQ(j["config"]["pp"]["lambda"].get<float>(), 0.3F);
}

/* Train with --vars uniform:count=2 saves vars config. */
TEST(CLITrain, TrainWithVarsStrategy) {
  TempFile const model;
  model.clear();
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 --vars uniform:count=2 -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_EQ(j["config"]["vars"]["name"], "uniform");
  EXPECT_EQ(j["config"]["vars"]["count"], 2);
}

/* Train with --vars all saves all vars config. */
TEST(CLITrain, TrainWithVarsAll) {
  TempFile const model;
  model.clear();
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 --vars all -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_EQ(j["config"]["vars"]["name"], "all");
}

/* Train with --cutpoint mean_of_means saves cutpoint config. */
TEST(CLITrain, TrainWithThresholdStrategy) {
  TempFile const model;
  model.clear();
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 --cutpoint mean_of_means -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_EQ(j["config"]["cutpoint"]["name"], "mean_of_means");
}

/* Train with all three explicit strategy flags produces correct config. */
TEST(CLITrain, TrainWithAllStrategies) {
  TempFile const model;
  model.clear();
  auto result = run_ppforest2(
      "-q train -d " + IRIS_CSV +
      " -n 5 -r 0 "
      "--pp pda:lambda=0.3 --vars uniform:count=2 --cutpoint mean_of_means "
      "-s " +
      model.path()
  );
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_EQ(j["config"]["pp"]["name"], "pda");
  EXPECT_FLOAT_EQ(j["config"]["pp"]["lambda"].get<float>(), 0.3F);
  EXPECT_EQ(j["config"]["vars"]["name"], "uniform");
  EXPECT_EQ(j["config"]["vars"]["count"], 2);
  EXPECT_EQ(j["config"]["cutpoint"]["name"], "mean_of_means");
}

/* Train with --leaf majority_vote saves leaf config. */
TEST(CLITrain, TrainWithLeafStrategy) {
  TempFile const model;
  model.clear();
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 --leaf majority_vote -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_EQ(j["config"]["leaf"]["name"], "majority_vote");
}

/* Default leaf strategy is majority_vote when --leaf is not specified. */
TEST(CLITrain, TrainDefaultLeafStrategy) {
  TempFile const model;
  model.clear();
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 -s " + model.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_EQ(j["config"]["leaf"]["name"], "majority_vote");
}

/* Invalid --leaf value fails. */
TEST(CLITrain, TrainInvalidLeafFails) {
  auto result = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 --leaf unknown_leaf --no-save");
  EXPECT_NE(result.exit_code, 0);
}

/* Strategy config does not contain display_name (presentation-only). */
TEST(CLITrain, StrategyConfigNoDisplayName) {
  TempFile const model;
  model.clear();
  auto result = run_ppforest2(
      "-q train -d " + IRIS_CSV +
      " -n 5 -r 0 "
      "--pp pda:lambda=0.3 --vars uniform:count=2 --cutpoint mean_of_means "
      "-s " +
      model.path()
  );
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(model.read());
  EXPECT_FALSE(j["config"]["pp"].contains("display_name"));
  EXPECT_FALSE(j["config"]["vars"].contains("display_name"));
  EXPECT_FALSE(j["config"]["cutpoint"].contains("display_name"));
}

// ---------------------------------------------------------------------------
// Implicit vs explicit strategy equivalence
// ---------------------------------------------------------------------------

/**
 * @brief Strip timing fields so two model exports can be compared.
 *
 * training_duration_ms varies between runs even with the same seed;
 * everything else (model, config, meta, metrics) must be identical.
 */
namespace {
  json strip_timing(json j) {
    j.erase("training_duration_ms");
    return j;
  }
}

/* -l 0.3 and --pp pda:lambda=0.3 produce identical exports. */
TEST(CLITrain, ImplicitExplicitPPEquivalent) {
  TempFile const model_implicit;
  TempFile const model_explicit;
  model_implicit.clear();
  model_explicit.clear();

  auto r1 = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 -l 0.3 -s " + model_implicit.path());
  auto r2 = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 --pp pda:lambda=0.3 -s " + model_explicit.path());
  ASSERT_EQ(r1.exit_code, 0);
  ASSERT_EQ(r2.exit_code, 0);

  EXPECT_EQ(strip_timing(json::parse(model_implicit.read())), strip_timing(json::parse(model_explicit.read())))
      << "Exports from -l and --pp should be identical";
}

/* --n-vars 2 and --vars uniform:count=2 produce identical exports. */
TEST(CLITrain, ImplicitExplicitVarsEquivalent) {
  TempFile const model_implicit;
  TempFile const model_explicit;
  model_implicit.clear();
  model_explicit.clear();

  auto r1 = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 --n-vars 2 -s " + model_implicit.path());
  auto r2 = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 --vars uniform:count=2 -s " + model_explicit.path());
  ASSERT_EQ(r1.exit_code, 0);
  ASSERT_EQ(r2.exit_code, 0);

  EXPECT_EQ(strip_timing(json::parse(model_implicit.read())), strip_timing(json::parse(model_explicit.read())))
      << "Exports from --n-vars and --vars should be identical";
}

/* -l 0.3 --n-vars 2 and --pp pda:lambda=0.3 --vars uniform:count=2 --cutpoint mean_of_means produce identical exports. */
TEST(CLITrain, ImplicitExplicitAllEquivalent) {
  TempFile const model_implicit;
  TempFile const model_explicit;
  model_implicit.clear();
  model_explicit.clear();

  auto r1 = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 -l 0.3 --n-vars 2 -s " + model_implicit.path());
  auto r2 = run_ppforest2(
      "-q train -d " + IRIS_CSV +
      " -n 5 -r 0 "
      "--pp pda:lambda=0.3 --vars uniform:count=2 --cutpoint mean_of_means -s " +
      model_explicit.path()
  );
  ASSERT_EQ(r1.exit_code, 0);
  ASSERT_EQ(r2.exit_code, 0);

  EXPECT_EQ(strip_timing(json::parse(model_implicit.read())), strip_timing(json::parse(model_explicit.read())))
      << "Exports from shortcuts and --pp/--vars/--cutpoint should be identical";
}

/* Single tree: -l 0.5 and --pp pda:lambda=0.5 --vars all --cutpoint mean_of_means produce identical exports. */
TEST(CLITrain, ImplicitExplicitSingleTreeEquivalent) {
  TempFile const model_implicit;
  TempFile const model_explicit;
  model_implicit.clear();
  model_explicit.clear();

  auto r1 = run_ppforest2("-q train -d " + IRIS_CSV + " -n 0 -r 0 -l 0.5 -s " + model_implicit.path());
  auto r2 = run_ppforest2(
      "-q train -d " + IRIS_CSV +
      " -n 0 -r 0 "
      "--pp pda:lambda=0.5 --vars all --cutpoint mean_of_means -s " +
      model_explicit.path()
  );
  ASSERT_EQ(r1.exit_code, 0);
  ASSERT_EQ(r2.exit_code, 0);

  EXPECT_EQ(strip_timing(json::parse(model_implicit.read())), strip_timing(json::parse(model_explicit.read())))
      << "Single tree with -l and --pp should be identical";
}

// ---------------------------------------------------------------------------
// Edge cases — degenerate inputs
// ---------------------------------------------------------------------------

namespace {
  using namespace ppforest2::types;

  TempFile write_csv(FeatureMatrix const& x, OutcomeVector const& y) {
    TempFile f;
    {
      std::ofstream out(f.path());

      for (int j = 0; j < x.cols(); ++j) {
        out << "\"x" << (j + 1) << "\",";
      }
      out << "\"group\"\n";

      for (int i = 0; i < x.rows(); ++i) {
        for (int j = 0; j < x.cols(); ++j) {
          out << x(i, j) << ",";
        }
        out << "\"g" << static_cast<int>(y(i)) << "\"\n";
      }
    }
    return f;
  }
}

TEST(CLITrainEdgeCase, ConstantFeatureColumn) {
  FeatureMatrix x = MAT(Feature, rows(6), 5, 1, 5, 2, 5, 3, 5, 7, 5, 8, 5, 9);
  OutcomeVector y = VEC(Outcome, 0, 0, 0, 1, 1, 1);

  auto csv    = write_csv(x, y);
  auto result = run_ppforest2("-q train -d " + csv.path() + " -n 5 -r 0 --no-save");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_EQ(result.stderr_output, "");
}

TEST(CLITrainEdgeCase, ConstantFeatureColumnSingleTree) {
  FeatureMatrix x = MAT(Feature, rows(6), 5, 1, 5, 2, 5, 3, 5, 7, 5, 8, 5, 9);
  OutcomeVector y = VEC(Outcome, 0, 0, 0, 1, 1, 1);

  auto csv    = write_csv(x, y);
  auto result = run_ppforest2("-q train -d " + csv.path() + " -n 0 --no-save");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_EQ(result.stderr_output, "");
}

TEST(CLITrainEdgeCase, SingleObservationPerGroup) {
  FeatureMatrix x = MAT(Feature, rows(2), 1, 0, 0, 1);
  OutcomeVector y = VEC(Outcome, 0, 1);

  auto csv    = write_csv(x, y);
  auto result = run_ppforest2("-q train -d " + csv.path() + " -n 5 -r 0 --no-save");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_EQ(result.stderr_output, "");
}

TEST(CLITrainEdgeCase, SingleObservationPerGroupSingleTree) {
  FeatureMatrix x = MAT(Feature, rows(2), 1, 0, 0, 1);
  OutcomeVector y = VEC(Outcome, 0, 1);

  auto csv    = write_csv(x, y);
  auto result = run_ppforest2("-q train -d " + csv.path() + " -n 0 --no-save");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_EQ(result.stderr_output, "");
}

TEST(CLITrainEdgeCase, ExtremeImbalance) {
  // clang-format off
  FeatureMatrix x = MAT(Feature, rows(20),
    0, 0,  1, 1,  2, 2,  3, 0,  4, 1,
    0, 2,  1, 0,  2, 1,  3, 2,  4, 0,
    0, 1,  1, 2,  2, 0,  3, 1,  4, 2,
    0, 0,  1, 1,  2, 2,  90, 90,  91, 91);
  OutcomeVector y = VEC(Outcome,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1);
  // clang-format on

  auto csv    = write_csv(x, y);
  auto result = run_ppforest2("-q train -d " + csv.path() + " -n 5 -r 0 --no-save");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_EQ(result.stderr_output, "");
}

TEST(CLITrainEdgeCase, ExtremeImbalanceSingleTree) {
  // clang-format off
  FeatureMatrix x = MAT(Feature, rows(20),
    0, 0,  1, 1,  2, 2,  3, 0,  4, 1,
    0, 2,  1, 0,  2, 1,  3, 2,  4, 0,
    0, 1,  1, 2,  2, 0,  3, 1,  4, 2,
    0, 0,  1, 1,  2, 2,  90, 90,  91, 91);
  OutcomeVector y = VEC(Outcome,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1);
  // clang-format on

  auto csv    = write_csv(x, y);
  auto result = run_ppforest2("-q train -d " + csv.path() + " -n 0 --no-save");
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_EQ(result.stderr_output, "");
}

// ---------------------------------------------------------------------------
// Strategy flags — predict pipeline
// ---------------------------------------------------------------------------

/* Model trained with --pp can be loaded by predict. */
TEST(CLITrain, TrainWithStrategyThenPredict) {
  TempFile const model;
  model.clear();
  auto train = run_ppforest2(
      "-q train -d " + IRIS_CSV +
      " -n 5 -r 0 "
      "--pp pda:lambda=0.3 --vars uniform:count=2 -s " +
      model.path()
  );
  ASSERT_EQ(train.exit_code, 0);

  TempFile const output;
  output.clear();
  auto predict = run_ppforest2("-q predict -M " + model.path() + " -d " + IRIS_CSV + " -o " + output.path());
  EXPECT_EQ(predict.exit_code, 0);

  auto j = json::parse(output.read());
  EXPECT_TRUE(j.contains("predictions"));
  EXPECT_GT(j["predictions"].size(), 0U);
}
