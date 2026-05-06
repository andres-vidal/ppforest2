/**
 * @file Reproducibility.test.cpp
 * @brief Tests that compare freshly-trained models against committed golden files.
 *
 * Each test loads a golden JSON file, trains the same model configuration,
 * and compares model structure, predictions, error rates, confusion matrix,
 * and (for forests) OOB error and variable importance.
 */
#include <gtest/gtest.h>

#include "utils/Types.hpp"
#include "stats/DataPacket.hpp"
#include "stats/Metrics.hpp"
#include "stats/Stats.hpp"
#include "stats/ConfusionMatrix.hpp"
#include "models/Model.hpp"
#include "models/ClassificationForest.hpp"
#include "models/Forest.hpp"
#include "models/Evaluation.hpp"
#include "models/TrainingSpec.hpp"
#include "serialization/Json.hpp"
#include "io/IO.hpp"
#include "utils/Invariant.hpp"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace ppforest2;
using namespace ppforest2::types;
using namespace ppforest2::stats;
using json = nlohmann::json;

#ifndef PPFOREST2_DATA_DIR
#error "PPFOREST2_DATA_DIR must be defined"
#endif

#ifndef PPFOREST2_GOLDEN_DIR
#error "PPFOREST2_GOLDEN_DIR must be defined"
#endif

#ifndef PPFOREST2_PLATFORM
#error "PPFOREST2_PLATFORM must be defined"
#endif

static const std::string DATA_DIR   = PPFOREST2_DATA_DIR;
static std::string const GOLDEN_DIR = PPFOREST2_GOLDEN_DIR;
static std::string const PLATFORM   = PPFOREST2_PLATFORM;

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

namespace {
  json load_golden(std::string const& path) {
    std::ifstream in(path);
    return json::parse(in);
  }

  /**
   * @brief Resolve the path to the golden file for a given dataset and slug.
   *
   * Tries to load the platform-specific golden file first, then falls back to
   * the base golden file.
   *
   * @param dataset The name of the dataset.
   * @param slug The slug of the golden file.
   * @return The path to the golden file.
   */
  std::string resolve_golden_path(std::string const& dataset, std::string const& slug) {
    std::string platform_path = GOLDEN_DIR + "/" + dataset + "/" + slug + "." + PLATFORM + ".json";

    if (std::filesystem::exists(platform_path)) {
      return platform_path;
    }

    return GOLDEN_DIR + "/" + dataset + "/" + slug + ".json";
  }

  void compare_predictions(
      json const& expected_json, OutcomeVector const& actual, std::vector<std::string> const& group_names
  ) {
    auto expected = expected_json.get<std::vector<std::string>>();

    ASSERT_EQ(static_cast<int>(expected.size()), actual.size()) << "Prediction count mismatch";

    for (int i = 0; i < actual.size(); ++i) {
      auto const& actual_label = group_names[static_cast<std::size_t>(actual(i))];
      EXPECT_EQ(expected[static_cast<std::size_t>(i)], actual_label) << "Prediction mismatch at index " << i;
    }
  }

  void compare_confusion_matrix(
      json const& expected_json, ConfusionMatrix const& actual, std::vector<std::string> const& group_names
  ) {
    auto const& expected_matrix = expected_json["matrix"];
    auto expected_labels        = expected_json["labels"].get<std::vector<std::string>>();

    ASSERT_EQ(static_cast<int>(expected_matrix.size()), actual.values.rows());

    for (int i = 0; i < actual.values.rows(); ++i) {
      for (int j = 0; j < actual.values.cols(); ++j) {
        EXPECT_EQ(expected_matrix[i][j].get<int>(), actual.values(i, j))
            << "Confusion matrix mismatch at (" << i << ", " << j << ")";
      }
    }

    // Compare labels
    std::vector<std::string> actual_labels;
    actual_labels.reserve(actual.label_index.size());
    for (auto const& [label, idx] : actual.label_index) {
      actual_labels.push_back(group_names[static_cast<std::size_t>(label)]);
    }

    ASSERT_EQ(expected_labels, actual_labels);
  }

  void compare_vi(json const& expected_json, std::string const& key, FeatureVector const& actual, float tolerance) {
    if (!expected_json.contains(key)) {
      return;
    }

    if (actual.size() == 0) {
      return;
    }

    auto expected = expected_json[key].get<std::vector<Feature>>();

    ASSERT_EQ(static_cast<int>(expected.size()), actual.size()) << "VI '" << key << "' size mismatch";

    for (int i = 0; i < actual.size(); ++i) {
      EXPECT_NEAR(expected[i], actual(i), tolerance) << "VI '" << key << "' mismatch at index " << i;
    }
  }

  void compare_strategy(json const& expected, json const& actual, std::string const& label) {
    for (auto const& [key, value] : expected.items()) {
      ASSERT_TRUE(actual.contains(key)) << label << ": missing key '" << key << "'";

      if (value.is_number_float()) {
        EXPECT_NEAR(value.get<Feature>(), actual[key].get<Feature>(), 1e-5F) << label << "." << key;
      } else {
        EXPECT_EQ(value, actual[key]) << label << "." << key;
      }
    }
  }

  void compare_config(json const& golden, TrainingSpec const& spec) {
    auto const& config = golden["config"];

    compare_strategy(config["pp"], spec.pp->to_json(), "pp");
    compare_strategy(config["vars"], spec.vars->to_json(), "vars");
    compare_strategy(config["cutpoint"], spec.cutpoint->to_json(), "cutpoint");
    compare_strategy(config["stop"], spec.stop->to_json(), "stop");
    compare_strategy(config["binarize"], spec.binarization->to_json(), "binarize");
    compare_strategy(config["grouping"], spec.grouping->to_json(), "grouping");

    EXPECT_EQ(config["size"].get<int>(), spec.size);
    EXPECT_EQ(config["seed"].get<int>(), spec.seed);
    EXPECT_EQ(config["max_retries"].get<int>(), spec.max_retries);
  }

  void compare_node(json const& expected, json const& actual, float tolerance, std::string const& path) {
    if (expected.contains("value")) {
      ASSERT_TRUE(actual.contains("value")) << path << ": expected leaf node";
      EXPECT_EQ(expected["value"].get<std::string>(), actual["value"].get<std::string>()) << path << ".value mismatch";
      return;
    }

    ASSERT_TRUE(actual.contains("projector")) << path << ": expected condition node";

    // Compare groups (string labels)
    auto expected_groups = expected["groups"].get<std::vector<std::string>>();
    auto actual_groups   = actual["groups"].get<std::vector<std::string>>();
    EXPECT_EQ(expected_groups, actual_groups) << path << ".groups mismatch";

    // Compare floats with tolerance
    EXPECT_NEAR(expected["pp_index_value"].get<Feature>(), actual["pp_index_value"].get<Feature>(), tolerance)
        << path << ".pp_index_value mismatch";

    EXPECT_NEAR(expected["cutpoint"].get<Feature>(), actual["cutpoint"].get<Feature>(), tolerance)
        << path << ".cutpoint mismatch";

    auto expected_proj = expected["projector"].get<std::vector<Feature>>();
    auto actual_proj   = actual["projector"].get<std::vector<Feature>>();
    ASSERT_EQ(expected_proj.size(), actual_proj.size()) << path << ".projector size mismatch";

    for (size_t i = 0; i < expected_proj.size(); ++i) {
      EXPECT_NEAR(expected_proj[i], actual_proj[i], tolerance) << path << ".projector[" << i << "] mismatch";
    }

    // Recurse into children
    compare_node(expected["lower"], actual["lower"], tolerance, path + ".lower");
    compare_node(expected["upper"], actual["upper"], tolerance, path + ".upper");
  }

  void compare_model_structure(
      json const& expected_json,
      Tree const& actual,
      std::vector<std::string> const& group_names,
      float tolerance = 1e-3F
  ) {
    json actual_json = serialization::to_json(actual, group_names);
    compare_node(expected_json["root"], actual_json["root"], tolerance, "root");
  }

  void compare_model_structure(
      json const& expected_json,
      Forest const& actual,
      std::vector<std::string> const& group_names,
      float tolerance = 1e-3F
  ) {
    json actual_json           = serialization::to_json(actual, group_names);
    auto const& expected_trees = expected_json["trees"];
    auto const& actual_trees   = actual_json["trees"];

    ASSERT_EQ(expected_trees.size(), actual_trees.size()) << "Tree count mismatch";

    for (size_t i = 0; i < expected_trees.size(); ++i) {
      compare_node(
          expected_trees[i]["root"], actual_trees[i]["root"], tolerance, "trees[" + std::to_string(i) + "].root"
      );
    }
  }

  void compare_vote_proportions(json const& expected_json, FeatureMatrix const& actual, float tolerance) {
    ASSERT_EQ(static_cast<int>(expected_json.size()), actual.rows()) << "Vote proportions row count mismatch";

    for (int i = 0; i < actual.rows(); ++i) {
      auto expected_row = expected_json[static_cast<std::size_t>(i)].get<std::vector<Feature>>();
      ASSERT_EQ(static_cast<int>(expected_row.size()), actual.cols())
          << "Vote proportions col count mismatch at row " << i;

      for (int j = 0; j < actual.cols(); ++j) {
        EXPECT_NEAR(expected_row[static_cast<std::size_t>(j)], actual(i, j), tolerance)
            << "Vote proportion mismatch at (" << i << ", " << j << ")";
      }
    }
  }
} // anonymous namespace

// ---------------------------------------------------------------------------
// Macro to reduce boilerplate for each golden test
// ---------------------------------------------------------------------------

#define GOLDEN_TREE_TEST(TestName, dataset_name, slug_name, csv_file, lambda_val, seed_val)                        \
  TEST(Reproducibility, TestName) {                                                                                \
    auto path = resolve_golden_path(dataset_name, slug_name);                                                      \
    invariant(std::filesystem::exists(path), "Golden file missing: " + path);                                      \
                                                                                                                   \
    auto golden      = load_golden(path);                                                                          \
    auto group_names = golden["meta"]["groups"].get<std::vector<std::string>>();                                   \
    auto data        = io::csv::read_sorted(DATA_DIR + "/" + csv_file);                                            \
                                                                                                                   \
    auto spec = TrainingSpec::builder(types::Mode::Classification).seed(seed_val).pp(pp::pda(lambda_val)).build(); \
    auto tree = Tree::train(spec, data.x, data.y);                                                                 \
                                                                                                                   \
                                                                                                                   \
    compare_model_structure(golden["model"], *tree, group_names);                                                  \
    compare_config(golden, *tree->training_spec);                                                                  \
                                                                                                                   \
    auto predictions = tree->predict(data.x);                                                                      \
    compare_predictions(golden["predictions"], predictions, group_names);                                          \
                                                                                                                   \
    GroupIdVector const y_int_tree = data.y.cast<GroupId>();                                                       \
    ConfusionMatrix cm(predictions.cast<GroupId>(), y_int_tree);                                                   \
    EXPECT_NEAR(cm.error(), golden["error_rate"].get<Feature>(), 1e-3F);                                           \
    compare_confusion_matrix(golden["training_metrics"]["confusion_matrix"], cm, group_names);                     \
                                                                                                                   \
    if (golden.contains("variable_importance")) {                                                                  \
      const int n_vars    = static_cast<int>(data.x.cols());                                                       \
      FeatureVector scale = stats::sd(data.x);                                                                     \
      scale               = (scale.array() > Feature(0)).select(scale, Feature(1));                                \
      FeatureVector vi2   = vi_projections(tree, n_vars, &scale);                                                  \
      compare_vi(golden["variable_importance"], "projections", vi2, 1e-3F);                                        \
    }                                                                                                              \
  }

#define GOLDEN_FOREST_TEST(TestName, dataset_name, slug_name, csv_file, n_trees, lambda_val, n_vars_val, seed_val) \
  TEST(Reproducibility, TestName) {                                                                                \
    auto path = resolve_golden_path(dataset_name, slug_name);                                                      \
    invariant(std::filesystem::exists(path), "Golden file missing: " + path);                                      \
                                                                                                                   \
    auto golden      = load_golden(path);                                                                          \
    auto group_names = golden["meta"]["groups"].get<std::vector<std::string>>();                                   \
    auto data        = io::csv::read_sorted(DATA_DIR + "/" + csv_file);                                            \
                                                                                                                   \
    auto forest = ClassificationForest::train(                                                                     \
        TrainingSpec::builder(types::Mode::Classification)                                                         \
            .size(n_trees)                                                                                         \
            .seed(seed_val)                                                                                        \
            .threads(1)                                                                                            \
            .pp(pp::pda(lambda_val))                                                                               \
            .vars(vars::uniform(n_vars_val))                                                                       \
            .build(),                                                                                              \
        data.x,                                                                                                    \
        data.y                                                                                                     \
    );                                                                                                             \
                                                                                                                   \
                                                                                                                   \
    compare_model_structure(golden["model"], *forest, group_names);                                                \
    compare_config(golden, *forest->training_spec);                                                                \
                                                                                                                   \
    auto predictions = forest->predict(data.x);                                                                    \
    compare_predictions(golden["predictions"], predictions, group_names);                                          \
                                                                                                                   \
    GroupIdVector const y_int_forest = data.y.cast<GroupId>();                                                     \
    ConfusionMatrix cm(predictions.cast<GroupId>(), y_int_forest);                                                 \
    EXPECT_NEAR(cm.error(), golden["error_rate"].get<Feature>(), 1e-3F);                                           \
    compare_confusion_matrix(golden["training_metrics"]["confusion_matrix"], cm, group_names);                     \
                                                                                                                   \
    if (golden.contains("oob_metrics") && !golden["oob_metrics"].is_null()) {                                      \
      auto oob_err = oob_error(forest, data.x, data.y);                                                            \
      ASSERT_TRUE(oob_err.has_value());                                                                            \
      EXPECT_NEAR(*oob_err, golden["oob_metrics"]["error_rate"].get<double>(), 1e-3);                              \
    }                                                                                                              \
                                                                                                                   \
    if (golden.contains("variable_importance")) {                                                                  \
      const int n_v       = static_cast<int>(data.x.cols());                                                       \
      FeatureVector scale = stats::sd(data.x);                                                                     \
      scale               = (scale.array() > Feature(0)).select(scale, Feature(1));                                \
                                                                                                                   \
      FeatureVector vi1 = vi_permuted(forest, data.x, data.y, seed_val);                                           \
      FeatureVector vi2 = vi_projections(forest, n_v, &scale);                                                     \
      FeatureVector vi3 = vi_weighted_projections(forest, data.x, data.y, &scale);                                 \
                                                                                                                   \
      compare_vi(golden["variable_importance"], "permuted", vi1, 1e-3F);                                           \
      compare_vi(golden["variable_importance"], "projections", vi2, 1e-3F);                                        \
      compare_vi(golden["variable_importance"], "weighted_projections", vi3, 1e-3F);                               \
    }                                                                                                              \
                                                                                                                   \
    if (golden.contains("vote_proportions")) {                                                                     \
      FeatureMatrix vote_props = forest->predict(data.x, Proportions{});                                           \
      compare_vote_proportions(golden["vote_proportions"], vote_props, 1e-3F);                                     \
    }                                                                                                              \
  }

// ---------------------------------------------------------------------------
// Tree tests
// ---------------------------------------------------------------------------

GOLDEN_TREE_TEST(IrisTreePDA, "iris", "tree-pda-s0", "classification/iris.csv", 0.0f, 0)
GOLDEN_TREE_TEST(CrabTreePDA, "crab", "tree-pda-s0", "classification/crab.csv", 0.0f, 0)

// No regression goldens: regression is experimental in v0.1.0 (see
// CHANGELOG). Pinning numerics would freeze the implementation at a
// stage we explicitly marked as "API and defaults may change." Non-
// golden regression tests (see `Regression.test.cpp`, `test-regression.R`,
// and the CLI integration tests) cover correctness; add golden
// reproducibility coverage when the regression pipeline stabilises.

// ---------------------------------------------------------------------------
// Forest tests
// ---------------------------------------------------------------------------

GOLDEN_FOREST_TEST(IrisForestPDAL0, "iris", "forest-pda-n5-s0", "classification/iris.csv", 5, 0.0F, 2, 0)
GOLDEN_FOREST_TEST(IrisForestPDAL05, "iris", "forest-pda-l05-n5-s0", "classification/iris.csv", 5, 0.5F, 2, 0)
GOLDEN_FOREST_TEST(CrabForestPDA, "crab", "forest-pda-n10-s0", "classification/crab.csv", 10, 0.0F, 3, 0)
GOLDEN_FOREST_TEST(WineForestPDA, "wine", "forest-pda-n10-s0", "classification/wine.csv", 10, 0.0F, 4, 0)
GOLDEN_FOREST_TEST(GlassForestPDA, "glass", "forest-pda-n10-s0", "classification/glass.csv", 10, 0.0F, 3, 0)
