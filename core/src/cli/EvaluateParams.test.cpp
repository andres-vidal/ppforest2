/**
 * @file EvaluateParams.test.cpp
 * @brief Unit tests for EvaluateParams and SimulateParams.
 */
#include <gtest/gtest.h>

#include <algorithm>

#include "cli/EvaluateParams.hpp"

using json = nlohmann::json;
using ppforest2::cli::EvaluateParams;
using ppforest2::cli::SimulateParams;

// ---------------------------------------------------------------------------
// EvaluateParams — constructor from JSON
// ---------------------------------------------------------------------------

TEST(EvaluateParams, DefaultConstruction) {
  EvaluateParams p;
  EXPECT_FALSE(p.train_ratio.has_value());
  EXPECT_FALSE(p.iterations.has_value());
  EXPECT_TRUE(p.export_path.empty());
  EXPECT_EQ(p.warmup, 0);
  EXPECT_FALSE(p.convergence.cv.has_value());
  EXPECT_FALSE(p.convergence.min.has_value());
  EXPECT_FALSE(p.convergence.max.has_value());
  EXPECT_FALSE(p.convergence.window.has_value());
}

TEST(EvaluateParams, ConstructFromJson) {
  json config = {
      {"train_ratio", 0.8},
      {"iterations", 50},
      {"warmup", 5},
      {"convergence", {{"cv", 0.03}, {"min", 20}, {"max", 500}, {"window", 4}}}
  };

  EvaluateParams p(config);
  EXPECT_FLOAT_EQ(*p.train_ratio, 0.8F);
  EXPECT_EQ(*p.iterations, 50);
  EXPECT_EQ(p.warmup, 5);
  EXPECT_FLOAT_EQ(*p.convergence.cv, 0.03F);
  EXPECT_EQ(*p.convergence.min, 20);
  EXPECT_EQ(*p.convergence.max, 500);
  EXPECT_EQ(*p.convergence.window, 4);
}

TEST(EvaluateParams, ConstructFromPartialJson) {
  json config = {{"train_ratio", 0.6}};

  EvaluateParams p(config);
  EXPECT_FLOAT_EQ(*p.train_ratio, 0.6F);
  EXPECT_FALSE(p.iterations.has_value());
  EXPECT_EQ(p.warmup, 0);
  EXPECT_FALSE(p.convergence.cv.has_value());
}

TEST(EvaluateParams, ConstructFromEmptyJson) {
  EvaluateParams p(json::object());
  EXPECT_FALSE(p.train_ratio.has_value());
  EXPECT_FALSE(p.iterations.has_value());
}

// ---------------------------------------------------------------------------
// EvaluateParams — convergence_enabled
// ---------------------------------------------------------------------------

TEST(EvaluateParams, ConvergenceEnabledByDefault) {
  EvaluateParams p;
  EXPECT_TRUE(p.convergence_enabled());
}

TEST(EvaluateParams, ConvergenceEnabledWhenIterationsZero) {
  EvaluateParams p;
  p.iterations = 0;
  EXPECT_TRUE(p.convergence_enabled());
}

TEST(EvaluateParams, ConvergenceEnabledWhenIterationsNegative) {
  EvaluateParams p;
  p.iterations = -1;
  EXPECT_TRUE(p.convergence_enabled());
}

TEST(EvaluateParams, ConvergenceDisabledWhenIterationsPositive) {
  EvaluateParams p;
  p.iterations = 10;
  EXPECT_FALSE(p.convergence_enabled());
}

// ---------------------------------------------------------------------------
// EvaluateParams — resolve_defaults
// ---------------------------------------------------------------------------

TEST(EvaluateParams, ResolveDefaultsFillsAllFields) {
  EvaluateParams p;
  p.resolve_defaults();

  EXPECT_FLOAT_EQ(*p.train_ratio, 0.7F);
  EXPECT_EQ(*p.iterations, 0);
  EXPECT_FLOAT_EQ(*p.convergence.cv, 0.05F);
  EXPECT_EQ(*p.convergence.min, 10);
  EXPECT_EQ(*p.convergence.max, 200);
  EXPECT_EQ(*p.convergence.window, 3);
}

TEST(EvaluateParams, ResolveDefaultsPreservesExplicitValues) {
  EvaluateParams p;
  p.train_ratio        = 0.8F;
  p.iterations         = 50;
  p.convergence.cv     = 0.01F;
  p.convergence.min    = 5;
  p.convergence.max    = 100;
  p.convergence.window = 2;

  p.resolve_defaults();

  EXPECT_FLOAT_EQ(*p.train_ratio, 0.8F);
  EXPECT_EQ(*p.iterations, 50);
  EXPECT_FLOAT_EQ(*p.convergence.cv, 0.01F);
  EXPECT_EQ(*p.convergence.min, 5);
  EXPECT_EQ(*p.convergence.max, 100);
  EXPECT_EQ(*p.convergence.window, 2);
}

TEST(EvaluateParams, ResolveDefaultsIsIdempotent) {
  EvaluateParams p;
  p.resolve_defaults();
  auto first = p.to_json();
  p.resolve_defaults();
  auto second = p.to_json();
  EXPECT_EQ(first, second);
}

// ---------------------------------------------------------------------------
// EvaluateParams — to_json
// ---------------------------------------------------------------------------

TEST(EvaluateParams, ToJsonAfterDefaults) {
  EvaluateParams p;
  p.resolve_defaults();

  auto j = p.to_json();
  EXPECT_FLOAT_EQ(j["train_ratio"].get<float>(), 0.7F);
  EXPECT_FALSE(j.contains("iterations"));
  EXPECT_EQ(j["warmup"].get<int>(), 0);
  EXPECT_FLOAT_EQ(j["convergence"]["cv"].get<float>(), 0.05F);
  EXPECT_EQ(j["convergence"]["min"].get<int>(), 10);
  EXPECT_EQ(j["convergence"]["max"].get<int>(), 200);
  EXPECT_EQ(j["convergence"]["window"].get<int>(), 3);
}

TEST(EvaluateParams, ToJsonWithIterations) {
  EvaluateParams p;
  p.iterations  = 50;
  p.train_ratio = 0.8F;

  auto j = p.to_json();
  EXPECT_EQ(j["iterations"].get<int>(), 50);
  EXPECT_FLOAT_EQ(j["train_ratio"].get<float>(), 0.8F);
}

TEST(EvaluateParams, ToJsonOmitsIterationsWhenZero) {
  EvaluateParams p;
  p.iterations = 0;

  auto j = p.to_json();
  EXPECT_FALSE(j.contains("iterations"));
}

TEST(EvaluateParams, ToJsonOmitsUnsetOptionals) {
  EvaluateParams p;
  auto j = p.to_json();
  EXPECT_FALSE(j.contains("train_ratio"));
  EXPECT_FALSE(j.contains("iterations"));
  EXPECT_FALSE(j.contains("convergence"));
}

// ---------------------------------------------------------------------------
// EvaluateParams — overrides
// ---------------------------------------------------------------------------

TEST(EvaluateParams, OverridesEmptyByDefault) {
  EvaluateParams p;
  auto j = p.overrides();
  EXPECT_TRUE(j.empty());
}

TEST(EvaluateParams, OverridesIncludesOnlySetFields) {
  EvaluateParams p;
  p.iterations     = 30;
  p.convergence.cv = 0.02F;

  auto j = p.overrides();
  EXPECT_EQ(j["iterations"].get<int>(), 30);
  EXPECT_FLOAT_EQ(j["convergence"]["cv"].get<float>(), 0.02F);
  EXPECT_FALSE(j.contains("train_ratio"));
  EXPECT_FALSE(j["convergence"].contains("min"));
}

TEST(EvaluateParams, OverridesEmptyAfterResolveDefaults) {
  // resolve_defaults fills all optionals, but overrides should
  // still reflect what was explicitly set before resolve
  EvaluateParams p;
  p.train_ratio = 0.6F;
  p.resolve_defaults();

  auto j = p.overrides();
  // After resolve, all fields are set, so overrides includes them all
  EXPECT_TRUE(j.contains("train_ratio"));
  EXPECT_TRUE(j.contains("iterations"));
  EXPECT_TRUE(j.contains("convergence"));
}

// ---------------------------------------------------------------------------
// EvaluateParams — JSON round-trip
// ---------------------------------------------------------------------------

TEST(EvaluateParams, JsonRoundTrip) {
  EvaluateParams original;
  original.train_ratio        = 0.8F;
  original.iterations         = 25;
  original.warmup             = 3;
  original.convergence.cv     = 0.02F;
  original.convergence.min    = 15;
  original.convergence.max    = 300;
  original.convergence.window = 5;

  auto j = original.to_json();
  EvaluateParams restored(j);

  EXPECT_FLOAT_EQ(*restored.train_ratio, *original.train_ratio);
  EXPECT_EQ(*restored.iterations, *original.iterations);
  EXPECT_EQ(restored.warmup, original.warmup);
  EXPECT_FLOAT_EQ(*restored.convergence.cv, *original.convergence.cv);
  EXPECT_EQ(*restored.convergence.min, *original.convergence.min);
  EXPECT_EQ(*restored.convergence.max, *original.convergence.max);
  EXPECT_EQ(*restored.convergence.window, *original.convergence.window);
}

// ---------------------------------------------------------------------------
// SimulateParams — constructor from JSON
// ---------------------------------------------------------------------------

TEST(SimulateParams, DefaultConstruction) {
  SimulateParams p;
  EXPECT_TRUE(p.format.empty());
  EXPECT_EQ(p.rows, 1000);
  EXPECT_EQ(p.cols, 10);
  EXPECT_EQ(p.n_groups, 2);
  EXPECT_FLOAT_EQ(p.classification.mean, 100.0F);
  EXPECT_FLOAT_EQ(p.classification.mean_separation, 50.0F);
  EXPECT_FLOAT_EQ(p.classification.sd, 10.0F);
  EXPECT_EQ(p.regression.n_informative, 0);
  EXPECT_FLOAT_EQ(p.regression.y_intercept, 0.0F);
  EXPECT_FLOAT_EQ(p.regression.y_sd, 0.1F);
  EXPECT_FLOAT_EQ(p.regression.sd, 1.0F);
}

TEST(SimulateParams, ConstructFromJson) {
  json config = {
      {"simulate", "200x5x3"},
      {"simulate_mean", 50.0},
      {"simulate_mean_separation", 25.0},
      {"simulate_sd", 5.0},
      {"simulate_n_informative", 3},
      {"simulate_y_intercept", 1.5},
      {"simulate_y_sd", 0.25},
  };

  SimulateParams p(config);
  EXPECT_EQ(p.format, "200x5x3");
  EXPECT_FLOAT_EQ(p.classification.mean, 50.0F);
  EXPECT_FLOAT_EQ(p.classification.mean_separation, 25.0F);
  EXPECT_FLOAT_EQ(p.classification.sd, 5.0F);
  EXPECT_FLOAT_EQ(p.regression.sd, 5.0F);
  EXPECT_EQ(p.regression.n_informative, 3);
  EXPECT_FLOAT_EQ(p.regression.y_intercept, 1.5F);
  EXPECT_FLOAT_EQ(p.regression.y_sd, 0.25F);
}

TEST(SimulateParams, ConstructFromEmptyJson) {
  SimulateParams p(json::object());
  EXPECT_TRUE(p.format.empty());
  EXPECT_FLOAT_EQ(p.classification.mean, 100.0F);
  EXPECT_FLOAT_EQ(p.regression.sd, 1.0F);
}

// ---------------------------------------------------------------------------
// SimulateParams — resolve_format
// ---------------------------------------------------------------------------

TEST(SimulateParams, ResolveFormatParsesNxPxG) {
  SimulateParams p;
  p.format = "500x20x4";
  p.resolve_format();

  EXPECT_EQ(p.rows, 500);
  EXPECT_EQ(p.cols, 20);
  EXPECT_EQ(p.n_groups, 4);
}

TEST(SimulateParams, ResolveFormatEmptyIsNoop) {
  SimulateParams p;
  p.resolve_format();

  EXPECT_EQ(p.rows, 1000);
  EXPECT_EQ(p.cols, 10);
  EXPECT_EQ(p.n_groups, 2);
}

TEST(SimulateParams, ResolveFormatMissingSecondX) {
  SimulateParams p;
  p.format = "500x20";
  p.resolve_format();

  // No second 'x' — format not parsed, defaults unchanged
  EXPECT_EQ(p.rows, 1000);
  EXPECT_EQ(p.cols, 10);
  EXPECT_EQ(p.n_groups, 2);
}

// ---------------------------------------------------------------------------
// EvaluateParams::validate
// ---------------------------------------------------------------------------

class ValidateEvaluateConfigTest : public ::testing::Test {
protected:
  static json base_config() {
    return {
        {"data", "iris.csv"},
        {"size", 100},
        {"lambda", 0.5},
        {"train_ratio", 0.7},
    };
  }

  std::vector<std::string> errors;

  bool has_error(std::string const& substring) {
    return std::any_of(errors.begin(), errors.end(), [&](auto const& e) {
      return e.find(substring) != std::string::npos;
    });
  }
};

TEST_F(ValidateEvaluateConfigTest, MissingTrainRatio) {
  json config = base_config();
  config.erase("train_ratio");

  EvaluateParams::validate(config, errors);

  EXPECT_TRUE(has_error("train_ratio"));
}

TEST_F(ValidateEvaluateConfigTest, InvalidTrainRatio) {
  json config           = base_config();
  config["train_ratio"] = 1.5;

  EvaluateParams::validate(config, errors);

  EXPECT_TRUE(has_error("train_ratio"));
}

TEST_F(ValidateEvaluateConfigTest, ValidConvergenceParams) {
  json config           = base_config();
  config["convergence"] = {{"cv", 0.05}, {"min", 10}, {"max", 200}, {"window", 3}};
  config["warmup"]      = 5;

  EvaluateParams::validate(config, errors);

  EXPECT_TRUE(errors.empty()) << errors[0];
}

TEST_F(ValidateEvaluateConfigTest, InvalidConvergenceCV) {
  json config           = base_config();
  config["convergence"] = {{"cv", 0.0}};

  EvaluateParams::validate(config, errors);

  EXPECT_TRUE(has_error("convergence.cv"));
}
