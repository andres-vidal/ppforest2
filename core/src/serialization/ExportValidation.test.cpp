#include <gtest/gtest.h>

#include "serialization/ExportValidation.hpp"
#include "test/ThrowsWith.hpp"

#include <filesystem>
#include <fstream>
#include <functional>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

using json = nlohmann::json;
using namespace ppforest2::serialization;
using ppforest2::test::throws_with;

#ifndef PPFOREST2_GOLDEN_DIR
#error "PPFOREST2_GOLDEN_DIR must be defined"
#endif

namespace {
  // Variant-agnostic test entry point: dispatch on `model_type` and run
  // the corresponding validator. Lets the rejection table stay flat — most
  // rows aren't variant-specific, and the few that are still pick
  // `validate_tree_export` / `validate_forest_export` directly.
  void validate_any(json const& j) {
    std::string const t = j.value("model_type", "");
    if (t == "forest") {
      validate_forest_export(j);
    } else {
      validate_tree_export(j);
    }
  }

  // A minimal, well-formed classification export skeleton. Enough to pass
  // validation — individual tests mutate one field to exercise rejections.
  json valid_classification_export() {
    return {
        {"model_type", "tree"},
        {"config",
         {{"mode", "classification"},
          {"size", 0},
          {"seed", 0},
          {"threads", 1},
          {"max_retries", 3},
          {"pp", {{"name", "pda"}, {"lambda", 0.0}}},
          {"vars", {{"name", "all"}}},
          {"cutpoint", {{"name", "mean_of_means"}}},
          {"stop", {{"name", "pure_node"}}},
          {"binarize", {{"name", "largest_gap"}}},
          {"grouping", {{"name", "by_label"}}},
          {"leaf", {{"name", "majority_vote"}}}}},
        {"meta",
         {{"observations", 30},
          {"features", 4},
          {"mode", "classification"},
          {"feature_names", {"a", "b", "c", "d"}},
          {"groups", {"x", "y"}}}},
        {"model", {{"root", {{"value", "x"}, {"degenerate", false}}}}}
    };
  }

  json valid_regression_export() {
    return {
        {"model_type", "forest"},
        {"config",
         {{"mode", "regression"},
          {"size", 5},
          {"seed", 0},
          {"threads", 1},
          {"max_retries", 3},
          {"pp", {{"name", "pda"}, {"lambda", 0.0}}},
          {"vars", {{"name", "all"}}},
          {"cutpoint", {{"name", "mean_of_means"}}},
          {"stop", {{"name", "min_size"}, {"min_size", 5}}},
          {"binarize", {{"name", "disabled"}}},
          {"grouping", {{"name", "by_cutpoint"}}},
          {"leaf", {{"name", "mean_response"}}}}},
        {"meta",
         {{"observations", 100},
          {"features", 3},
          {"mode", "regression"},
          {"feature_names", json::array()},
          {"groups", json::array()}}},
        {"model", {{"trees", json::array()}}}
    };
  }
}

// -----------------------------------------------------------------------
// Happy paths — valid exports must pass every applicable validator.
// -----------------------------------------------------------------------

TEST(ExportValidation, AcceptsValidClassification) {
  EXPECT_NO_THROW(validate_tree_export(valid_classification_export()));
  EXPECT_NO_THROW(validate_any(valid_classification_export()));
}

TEST(ExportValidation, AcceptsValidRegression) {
  EXPECT_NO_THROW(validate_forest_export(valid_regression_export()));
  EXPECT_NO_THROW(validate_any(valid_regression_export()));
}

TEST(ExportValidation, AcceptsUnknownTopLevelKey) {
  // Top level is deliberately extensible — downstream tools (CLI, golden
  // generator) add their own annotations. Only required keys and known
  // fields are validated; extras pass through.
  json j        = valid_classification_export();
  j["surprise"] = 1;
  EXPECT_NO_THROW(validate_any(j));
}

// -----------------------------------------------------------------------
// Rejection table
//
// Every row: a base fixture, a single mutation, the validator under
// test, and the substring expected in the thrown exception's message.
// Adding a rejection case is a one-line table extension — no new TEST.
// -----------------------------------------------------------------------

namespace {
  using BaseFn    = json (*)();
  using Validator = void (*)(json const&);
  using Mutator   = std::function<void(json&)>;

  struct Rejection {
    char const* name;
    BaseFn base;
    Mutator mutate;
    Validator validate;
    char const* needle;
  };

  class ExportValidationRejects : public ::testing::TestWithParam<Rejection> {};

  TEST_P(ExportValidationRejects, Rejects) {
    auto const& c = GetParam();
    json j        = c.base();
    c.mutate(j);
    EXPECT_TRUE(throws_with([&] { c.validate(j); }, c.needle));
  }

  std::vector<Rejection> const REJECTIONS = {
      // ---- Top-level skeleton ----
      {"MissingConfig",
       valid_classification_export,
       [](json& j) { j.erase("config"); },
       validate_any,
       "Export.config: missing"},
      {"MissingMeta",
       valid_classification_export,
       [](json& j) { j.erase("meta"); },
       validate_any,
       "Export.meta: missing"},
      {"MissingModel",
       valid_classification_export,
       [](json& j) { j.erase("model"); },
       validate_any,
       "Export.model: missing"},
      {"BogusModelType",
       valid_classification_export,
       [](json& j) { j["model_type"] = "random_forest"; },
       validate_any,
       "Export.model_type"},
      {"MissingModelType",
       valid_classification_export,
       [](json& j) { j.erase("model_type"); },
       validate_any,
       "Export.model_type: missing"},

      // ---- config block ----
      {"BogusMode",
       valid_classification_export,
       [](json& j) { j["config"]["mode"] = "regresion"; },
       validate_any,
       "Export.config.mode: must be one of"},
      {"NegativeSize",
       valid_classification_export,
       [](json& j) { j["config"]["size"] = -1; },
       validate_any,
       "Export.config.size"},
      // Regression specs carry `binarize::Disabled` rather than omitting the
      // key — validation requires `binarize` to be present for every mode.
      {"MissingBinarizeForClassification",
       valid_classification_export,
       [](json& j) { j["config"].erase("binarize"); },
       validate_any,
       "Export.config.binarize: missing"},
      {"MissingBinarizeForRegression",
       valid_regression_export,
       [](json& j) { j["config"].erase("binarize"); },
       validate_any,
       "Export.config.binarize: missing"},

      // ---- meta block (groups required for both modes; non-empty only for classification) ----
      {"ClassificationWithoutGroups",
       valid_classification_export,
       [](json& j) { j["meta"].erase("groups"); },
       validate_any,
       "Export.meta.groups: missing"},
      {"ClassificationWithEmptyGroups",
       valid_classification_export,
       [](json& j) { j["meta"]["groups"] = json::array(); },
       validate_any,
       "Export.meta.groups: must be non-empty"},
      {"RegressionWithoutGroups",
       valid_regression_export,
       [](json& j) { j["meta"].erase("groups"); },
       validate_any,
       "Export.meta.groups: missing"},
      {"NonStringGroupEntry",
       valid_classification_export,
       [](json& j) { j["meta"]["groups"] = json::array({"x", 42}); },
       validate_any,
       "Export.meta.groups[1]"},
      {"NegativeObservations",
       valid_classification_export,
       [](json& j) { j["meta"]["observations"] = -3; },
       validate_any,
       "Export.meta.observations"},
      {"MissingFeatureNames",
       valid_classification_export,
       [](json& j) { j["meta"].erase("feature_names"); },
       validate_any,
       "Export.meta.feature_names: missing"},

      // ---- Variant-specific asserts ----
      {"TreeRejectsForestModelType",
       valid_classification_export,
       [](json& j) { j["model_type"] = "forest"; },
       validate_tree_export,
       "expected 'tree'"},
      {"ForestRejectsTreeModelType",
       valid_regression_export,
       [](json& j) { j["model_type"] = "tree"; },
       validate_forest_export,
       "expected 'forest'"},
      {"TreeMissingRoot",
       valid_classification_export,
       [](json& j) { j["model"] = json::object(); },
       validate_tree_export,
       "Export.model.root: missing"},
      {"ForestNonArrayTrees",
       valid_regression_export,
       [](json& j) { j["model"]["trees"] = 7; },
       validate_forest_export,
       "Export.model.trees"},

      // ---- Variant-agnostic dispatch via validate_any ----
      // model_type drives which model-block check runs.
      {"ModelDispatchTreeMissingRoot",
       valid_classification_export,
       [](json& j) { j["model"] = json::object(); },
       validate_any,
       "Export.model.root: missing"},
      {"ModelDispatchForestNonArrayTrees",
       valid_regression_export,
       [](json& j) { j["model"]["trees"] = 7; },
       validate_any,
       "Export.model.trees"},
  };

  INSTANTIATE_TEST_SUITE_P(Cases, ExportValidationRejects, ::testing::ValuesIn(REJECTIONS), [](auto const& info) {
    return info.param.name;
  });
}

// -----------------------------------------------------------------------
// Drift protection: every committed golden file must validate.
// -----------------------------------------------------------------------

TEST(ExportValidation, AllGoldenFilesValidate) {
  std::filesystem::path const golden_dir = PPFOREST2_GOLDEN_DIR;
  ASSERT_TRUE(std::filesystem::is_directory(golden_dir)) << golden_dir;

  int checked = 0;
  for (auto const& entry : std::filesystem::recursive_directory_iterator(golden_dir)) {
    if (!entry.is_regular_file() || entry.path().extension() != ".json")
      continue;

    std::ifstream in(entry.path());
    ASSERT_TRUE(in.is_open()) << entry.path();

    json j;
    in >> j;

    // Variant-agnostic validator — accepts both `model_type=tree` and
    // `model_type=forest` goldens.
    EXPECT_NO_THROW(validate_any(j)) << "golden file failed validation: " << entry.path();
    ++checked;
  }

  ASSERT_GT(checked, 0) << "no golden files found under " << golden_dir;
}
