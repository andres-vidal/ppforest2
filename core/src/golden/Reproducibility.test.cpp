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
#include "utils/Math.hpp"
#include "stats/DataPacket.hpp"
#include "stats/Stats.hpp"
#include "stats/ConfusionMatrix.hpp"
#include "models/Tree.hpp"
#include "models/Forest.hpp"
#include "models/TrainingSpecPDA.hpp"
#include "models/TrainingSpecUPDA.hpp"
#include "models/VariableImportance.hpp"
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
static const std::string GOLDEN_DIR = PPFOREST2_GOLDEN_DIR;
static const std::string PLATFORM   = PPFOREST2_PLATFORM;

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

namespace {
  json load_golden(const std::string& path) {
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
  std::string resolve_golden_path(const std::string& dataset, const std::string& slug) {
    std::string platform_path = GOLDEN_DIR + "/" + dataset + "/" + slug + "." + PLATFORM + ".json";

    if (std::filesystem::exists(platform_path)) {
      return platform_path;
    }

    return GOLDEN_DIR + "/" + dataset + "/" + slug + ".json";
  }

  void compare_predictions(const json& expected_json, const ResponseVector& actual,
    const std::vector<std::string>& group_names) {
    auto expected = expected_json.get<std::vector<std::string>>();

    ASSERT_EQ(static_cast<int>(expected.size()), actual.size()) << "Prediction count mismatch";

    for (int i = 0; i < actual.size(); ++i) {
      std::string actual_label = group_names[static_cast<std::size_t>(actual(i))];
      EXPECT_EQ(expected[static_cast<std::size_t>(i)], actual_label) << "Prediction mismatch at index " << i;
    }
  }

  void compare_confusion_matrix(const json& expected_json, const ConfusionMatrix& actual,
    const std::vector<std::string>& group_names) {
    auto expected_matrix = expected_json["matrix"];
    auto expected_labels = expected_json["labels"].get<std::vector<std::string>>();

    ASSERT_EQ(static_cast<int>(expected_matrix.size()), actual.values.rows());

    for (int i = 0; i < actual.values.rows(); ++i) {
      for (int j = 0; j < actual.values.cols(); ++j) {
        EXPECT_EQ(expected_matrix[i][j].get<int>(), actual.values(i, j)) << "Confusion matrix mismatch at (" << i << ", " << j << ")";
      }
    }

    // Compare labels
    std::vector<std::string> actual_labels;
    for (const auto& [label, idx] : actual.label_index) {
      actual_labels.push_back(group_names[static_cast<std::size_t>(label)]);
    }

    ASSERT_EQ(expected_labels, actual_labels);
  }

  void compare_vi(const json& expected_json, const std::string& key,
    const FeatureVector& actual, float tolerance) {
    if (!expected_json.contains(key)) return;

    if (actual.size() == 0) return;

    auto expected = expected_json[key].get<std::vector<float>>();

    ASSERT_EQ(static_cast<int>(expected.size()), actual.size()) << "VI '" << key << "' size mismatch";

    for (int i = 0; i < actual.size(); ++i) {
      EXPECT_NEAR(expected[i], actual(i), tolerance) << "VI '" << key << "' mismatch at index " << i;
    }
  }

  void compare_node(const json& expected, const json& actual,
    float tolerance, const std::string& path) {
    if (expected.contains("value")) {
      ASSERT_TRUE(actual.contains("value")) << path << ": expected leaf node";
      EXPECT_EQ(expected["value"].get<std::string>(), actual["value"].get<std::string>())
        << path << ".value mismatch";
      return;
    }

    ASSERT_TRUE(actual.contains("projector")) << path << ": expected condition node";

    // Compare groups (string labels)
    auto expected_groups = expected["groups"].get<std::vector<std::string>>();
    auto actual_groups   = actual["groups"].get<std::vector<std::string>>();
    EXPECT_EQ(expected_groups, actual_groups) << path << ".groups mismatch";

    // Compare floats with tolerance
    EXPECT_NEAR(expected["pp_index_value"].get<float>(),
      actual["pp_index_value"].get<float>(), tolerance)
      << path << ".pp_index_value mismatch";

    EXPECT_NEAR(expected["threshold"].get<float>(),
      actual["threshold"].get<float>(), tolerance)
      << path << ".threshold mismatch";

    auto expected_proj = expected["projector"].get<std::vector<float>>();
    auto actual_proj   = actual["projector"].get<std::vector<float>>();
    ASSERT_EQ(expected_proj.size(), actual_proj.size())
      << path << ".projector size mismatch";

    for (size_t i = 0; i < expected_proj.size(); ++i) {
      EXPECT_NEAR(expected_proj[i], actual_proj[i], tolerance)
        << path << ".projector[" << i << "] mismatch";
    }

    // Recurse into children
    compare_node(expected["lower"], actual["lower"], tolerance, path + ".lower");
    compare_node(expected["upper"], actual["upper"], tolerance, path + ".upper");
  }

  void compare_model_structure(const json& expected_json, const Tree& actual,
    const std::vector<std::string>& group_names, float tolerance = 1e-3f) {
    json actual_json = serialization::to_json(actual, group_names);
    compare_node(expected_json["root"], actual_json["root"], tolerance, "root");
  }

  void compare_model_structure(const json& expected_json, const Forest& actual,
    const std::vector<std::string>& group_names, float tolerance = 1e-3f) {
    json actual_json     = serialization::to_json(actual, group_names);
    auto& expected_trees = expected_json["trees"];
    auto& actual_trees   = actual_json["trees"];

    ASSERT_EQ(expected_trees.size(), actual_trees.size()) << "Tree count mismatch";

    for (size_t i = 0; i < expected_trees.size(); ++i) {
      compare_node(expected_trees[i]["root"], actual_trees[i]["root"],
        tolerance, "trees[" + std::to_string(i) + "].root");
    }
  }

  void compare_vote_proportions(const json& expected_json, const FeatureMatrix& actual, float tolerance) {
    ASSERT_EQ(static_cast<int>(expected_json.size()), actual.rows()) << "Vote proportions row count mismatch";

    for (int i = 0; i < actual.rows(); ++i) {
      auto expected_row = expected_json[static_cast<std::size_t>(i)].get<std::vector<float>>();
      ASSERT_EQ(static_cast<int>(expected_row.size()), actual.cols()) << "Vote proportions col count mismatch at row " << i;

      for (int j = 0; j < actual.cols(); ++j) {
        EXPECT_NEAR(expected_row[static_cast<std::size_t>(j)], actual(i, j), tolerance) << "Vote proportion mismatch at (" << i << ", " << j << ")";
      }
    }
  }
} // anonymous namespace

// ---------------------------------------------------------------------------
// Macro to reduce boilerplate for each golden test
// ---------------------------------------------------------------------------

#define GOLDEN_TREE_TEST(TestName, dataset_name, slug_name, csv_file, lambda_val, seed_val) \
        TEST(Reproducibility, TestName) {                                                   \
          auto path = resolve_golden_path(dataset_name, slug_name);                         \
          invariant(std::filesystem::exists(path), "Golden file missing: " + path);         \
                                                                                            \
          auto golden      = load_golden(path);                                             \
          auto group_names = golden["meta"]["groups"].get<std::vector<std::string>>();      \
          auto data        = io::csv::read_sorted(DATA_DIR + "/" + csv_file);               \
                                                                                            \
          RNG rng(seed_val);                                                                \
          Tree tree = Tree::train(TrainingSpecPDA(lambda_val), data.x, data.y, rng);        \
                                                                                            \
          compare_model_structure(golden["model"], tree, group_names);                      \
                                                                                            \
          auto predictions = tree.predict(data.x);                                          \
          compare_predictions(golden["predictions"], predictions, group_names);             \
                                                                                            \
          ConfusionMatrix cm(predictions, data.y);                                          \
          EXPECT_NEAR(cm.error(), golden["error_rate"].get<float>(), 1e-3f);                \
          compare_confusion_matrix(golden["training_confusion_matrix"], cm, group_names);   \
                                                                                            \
          if (golden.contains("variable_importance")) {                                     \
            const int n_vars    = static_cast<int>(data.x.cols());                          \
            FeatureVector scale = stats::sd(data.x);                                        \
            scale = (scale.array() > Feature(0)).select(scale, Feature(1));                 \
            FeatureVector vi2 = variable_importance_projections(tree, n_vars, &scale);      \
            compare_vi(golden["variable_importance"], "projections", vi2, 1e-3f);           \
          }                                                                                 \
        }

#define GOLDEN_FOREST_TEST(TestName, dataset_name, slug_name, csv_file,                         \
          n_trees, lambda_val, n_vars_val, seed_val)                                            \
        TEST(Reproducibility, TestName) {                                                       \
          auto path = resolve_golden_path(dataset_name, slug_name);                             \
          invariant(std::filesystem::exists(path), "Golden file missing: " + path);             \
                                                                                                \
          auto golden      = load_golden(path);                                                 \
          auto group_names = golden["meta"]["groups"].get<std::vector<std::string>>();          \
          auto data        = io::csv::read_sorted(DATA_DIR + "/" + csv_file);                   \
                                                                                                \
          Forest forest = Forest::train(                                                        \
            TrainingSpecUPDA(n_vars_val, lambda_val),                                           \
            data.x, data.y, n_trees, seed_val, 1);                                              \
                                                                                                \
          compare_model_structure(golden["model"], forest, group_names);                        \
                                                                                                \
          auto predictions = forest.predict(data.x);                                            \
          compare_predictions(golden["predictions"], predictions, group_names);                 \
                                                                                                \
          ConfusionMatrix cm(predictions, data.y);                                              \
          EXPECT_NEAR(cm.error(), golden["error_rate"].get<float>(), 1e-3f);                    \
          compare_confusion_matrix(golden["training_confusion_matrix"], cm, group_names);       \
                                                                                                \
          if (golden.contains("oob_error")) {                                                   \
            double oob_err = forest.oob_error(data.x, data.y);                                  \
            EXPECT_NEAR(oob_err, golden["oob_error"].get<double>(), 1e-3);                      \
          }                                                                                     \
                                                                                                \
          if (golden.contains("variable_importance")) {                                         \
            const int n_v       = static_cast<int>(data.x.cols());                              \
            FeatureVector scale = stats::sd(data.x);                                            \
            scale = (scale.array() > Feature(0)).select(scale, Feature(1));                     \
                                                                                                \
            FeatureVector vi1 = variable_importance_permuted(forest, data.x, data.y, seed_val); \
            FeatureVector vi2 = variable_importance_projections(forest, n_v, &scale);           \
            FeatureVector vi3 = variable_importance_weighted_projections(                       \
              forest, data.x, data.y, &scale);                                                  \
                                                                                                \
            compare_vi(golden["variable_importance"], "permuted", vi1, 1e-3f);                  \
            compare_vi(golden["variable_importance"], "projections", vi2, 1e-3f);               \
            compare_vi(golden["variable_importance"], "weighted_projections", vi3, 1e-3f);      \
          }                                                                                     \
                                                                                                \
          if (golden.contains("vote_proportions")) {                                            \
            FeatureMatrix vote_props = forest.predict(data.x, Proportions{});                   \
            compare_vote_proportions(golden["vote_proportions"], vote_props, 1e-3f);            \
          }                                                                                     \
        }

// ---------------------------------------------------------------------------
// Tree tests
// ---------------------------------------------------------------------------

GOLDEN_TREE_TEST(IrisTreePDA, "iris", "tree-pda-s42", "iris.csv", 0.0f, 42)
GOLDEN_TREE_TEST(CrabTreePDA, "crab", "tree-pda-s42", "crab.csv", 0.0f, 42)

// ---------------------------------------------------------------------------
// Forest tests
// ---------------------------------------------------------------------------

GOLDEN_FOREST_TEST(IrisForestPDAL0, "iris", "forest-pda-t5-s42", "iris.csv", 5, 0.0f, 2, 42)
GOLDEN_FOREST_TEST(IrisForestPDAL05, "iris", "forest-pda-l05-t5-s42", "iris.csv", 5, 0.5f, 2, 42)
GOLDEN_FOREST_TEST(CrabForestPDA, "crab", "forest-pda-t10-s42", "crab.csv", 10, 0.0f, 3, 42)
GOLDEN_FOREST_TEST(WineForestPDA, "wine", "forest-pda-t10-s42", "wine.csv", 10, 0.0f, 4, 42)
GOLDEN_FOREST_TEST(GlassForestPDA, "glass", "forest-pda-t10-s42", "glass.csv", 10, 0.0f, 3, 42)
