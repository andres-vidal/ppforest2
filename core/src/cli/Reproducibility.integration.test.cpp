/**
 * @file Reproducibility.integration.test.cpp
 * @brief CLI golden tests — end-to-end reproducibility through CLI pipeline.
 */
#include "cli/CLI.integration.hpp"

// ---------------------------------------------------------------------------
// Golden test helpers — load reference files and compare fields
// ---------------------------------------------------------------------------

static json load_golden(const std::string& dataset, const std::string& slug) {
  std::string path = GOLDEN_DIR + "/" + dataset + "/" + slug + ".json";
  std::ifstream in(path);
  EXPECT_TRUE(in.good()) << "Golden file not found: " << path;
  return json::parse(in);
}

static void compare_predictions(const json& actual, const json& expected) {
  ASSERT_EQ(actual.size(), expected.size()) << "predictions size mismatch";

  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(actual[i], expected[i]) << "prediction[" << i << "]";
  }
}

static void compare_float_array(
  const json& actual, const json& expected, double tol, const std::string& name) {
  ASSERT_EQ(actual.size(), expected.size()) << name << " size mismatch";
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(actual[i].get<double>(), expected[i].get<double>(), tol)
      << name << "[" << i << "]";
  }
}

static void compare_confusion_matrix(const json& actual, const json& expected) {
  EXPECT_EQ(actual["matrix"], expected["matrix"]) << "confusion matrix mismatch";
  EXPECT_EQ(actual["labels"], expected["labels"]) << "confusion labels mismatch";
}

// ---------------------------------------------------------------------------
// Golden test macros — train via CLI, predict, compare against golden files
// ---------------------------------------------------------------------------

#define CLI_GOLDEN_TREE_TEST(TestName, dataset, slug, csv, lambda, seed)                                                           \
        TEST(CLIGolden, TestName) {                                                                                                \
          auto golden = load_golden(dataset, slug);                                                                                \
                                                                                                                                   \
          TempFile model;                                                                                                          \
          model.clear();                                                                                                           \
          auto train = run_ppforest2(                                                                                              \
            "-q train -d " + csv + " -t 0"                                                                                         \
            " -l " + std::to_string(lambda) +                                                                                      \
            " -r " + std::to_string(seed) +                                                                                        \
            " -s " + model.path());                                                                                                \
          ASSERT_EQ(train.exit_code, 0) << "train failed";                                                                         \
                                                                                                                                   \
          auto model_json = json::parse(model.read());                                                                             \
          ASSERT_TRUE(model_json.contains("variable_importance"));                                                                 \
          auto vi = model_json["variable_importance"];                                                                             \
          compare_float_array(vi["scale"], golden["variable_importance"]["scale"], 1e-3, #TestName " VI scale");                   \
          compare_float_array(vi["projections"], golden["variable_importance"]["projections"], 1e-3, #TestName " VI projections"); \
                                                                                                                                   \
          TempFile output;                                                                                                         \
          output.clear();                                                                                                          \
          auto predict = run_ppforest2("-q predict -M " + model.path() + " -d " + csv + " -o " + output.path());                   \
          ASSERT_EQ(predict.exit_code, 0) << "predict failed";                                                                     \
                                                                                                                                   \
          auto pred_json = json::parse(output.read());                                                                             \
          compare_predictions(pred_json["predictions"], golden["predictions"]);                                                    \
          EXPECT_NEAR(pred_json["error_rate"].get<double>(), golden["error_rate"].get<double>(), 1e-3) << #TestName " error_rate"; \
          compare_confusion_matrix(pred_json["confusion_matrix"], golden["training_confusion_matrix"]);                            \
        }

#define CLI_GOLDEN_FOREST_TEST(TestName, dataset, slug, csv, n_trees, lambda, n_vars, seed)                                         \
        TEST(CLIGolden, TestName) {                                                                                                 \
          auto golden = load_golden(dataset, slug);                                                                                 \
                                                                                                                                    \
          TempFile model;                                                                                                           \
          model.clear();                                                                                                            \
          auto train = run_ppforest2(                                                                                               \
            "-q train -d " + csv +                                                                                                  \
            " -t " + std::to_string(n_trees) +                                                                                      \
            " -l " + std::to_string(lambda) +                                                                                       \
            " -r " + std::to_string(seed) +                                                                                         \
            " -v " + std::to_string(n_vars) +                                                                                       \
            " --threads 1"                                                                                                          \
            " -s " + model.path());                                                                                                 \
          ASSERT_EQ(train.exit_code, 0) << "train failed";                                                                          \
                                                                                                                                    \
          auto model_json = json::parse(model.read());                                                                              \
          ASSERT_TRUE(model_json.contains("oob_error"));                                                                            \
          EXPECT_NEAR(model_json["oob_error"].get<double>(), golden["oob_error"].get<double>(), 1e-3) << #TestName " oob_error";    \
                                                                                                                                    \
          ASSERT_TRUE(model_json.contains("variable_importance"));                                                                  \
          auto vi  = model_json["variable_importance"];                                                                             \
          auto gvi = golden["variable_importance"];                                                                                 \
          compare_float_array(vi["scale"], gvi["scale"], 1e-3, #TestName " VI scale");                                              \
          compare_float_array(vi["projections"], gvi["projections"], 1e-3, #TestName " VI projections");                            \
          compare_float_array(vi["weighted_projections"], gvi["weighted_projections"], 1e-3, #TestName " VI weighted_projections"); \
          compare_float_array(vi["permuted"], gvi["permuted"], 1e-3, #TestName " VI permuted");                                     \
                                                                                                                                    \
          TempFile output;                                                                                                          \
          output.clear();                                                                                                           \
          auto predict = run_ppforest2("-q predict -M " + model.path() + " -d " + csv + " -o " + output.path());                    \
          ASSERT_EQ(predict.exit_code, 0) << "predict failed";                                                                      \
                                                                                                                                    \
          auto pred_json = json::parse(output.read());                                                                              \
          compare_predictions(pred_json["predictions"], golden["predictions"]);                                                     \
          EXPECT_NEAR(pred_json["error_rate"].get<double>(), golden["error_rate"].get<double>(), 1e-3) << #TestName " error_rate";  \
          compare_confusion_matrix(pred_json["confusion_matrix"], golden["training_confusion_matrix"]);                             \
        }

// ---------------------------------------------------------------------------
// Golden tests
// ---------------------------------------------------------------------------

CLI_GOLDEN_TREE_TEST(IrisTreePDA,   "iris", "tree-pda-s42",   IRIS_CSV, 0.0f, 42)
CLI_GOLDEN_TREE_TEST(CrabTreePDA,   "crab", "tree-pda-s42",   CRAB_CSV, 0.0f, 42)

CLI_GOLDEN_FOREST_TEST(IrisForestPDAL0,  "iris",  "forest-pda-t5-s42",      IRIS_CSV,  5,  0.0f, 2, 42)
CLI_GOLDEN_FOREST_TEST(IrisForestPDAL05, "iris",  "forest-pda-l05-t5-s42",   IRIS_CSV,  5,  0.5f, 2, 42)
CLI_GOLDEN_FOREST_TEST(CrabForestPDA,  "crab",  "forest-pda-t10-s42",     CRAB_CSV,  10, 0.0f, 3, 42)
CLI_GOLDEN_FOREST_TEST(WineForestPDA,  "wine",  "forest-pda-t10-s42",     WINE_CSV,  10, 0.0f, 4, 42)
CLI_GOLDEN_FOREST_TEST(GlassForestPDA, "glass", "forest-pda-t10-s42",     GLASS_CSV, 10, 0.0f, 3, 42)
