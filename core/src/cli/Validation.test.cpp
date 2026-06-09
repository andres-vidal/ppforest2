/**
 * @file Validation.test.cpp
 * @brief Tests for central config validation, data source checks, and defaults.
 */
#include <gtest/gtest.h>
#include "cli/Validation.hpp"
#include "cli/CLIOptions.hpp"

using namespace ppforest2::cli;
using json = nlohmann::json;

class ValidationDefaultsTest : public ::testing::Test {};

TEST_F(ValidationDefaultsTest, TrainingDefaultsContainsExpectedKeys) {
  auto defaults = training_defaults();

  EXPECT_TRUE(defaults.contains("size"));
  EXPECT_TRUE(defaults.contains("lambda"));
  EXPECT_TRUE(defaults.contains("train_ratio"));
  EXPECT_TRUE(defaults.contains("max_retries"));
  // seed and threads are std::optional — not included in defaults
  EXPECT_FALSE(defaults.contains("seed"));
  EXPECT_FALSE(defaults.contains("threads"));
}

TEST_F(ValidationDefaultsTest, TrainingDefaultValues) {
  auto defaults = training_defaults();

  EXPECT_EQ(defaults["size"], 100);
  EXPECT_FLOAT_EQ(defaults["lambda"].get<float>(), 0.0F);
  EXPECT_FLOAT_EQ(defaults["train_ratio"].get<float>(), 0.7F);
  EXPECT_FALSE(defaults.contains("seed"));
  EXPECT_FALSE(defaults.contains("threads"));
  EXPECT_EQ(defaults["max_retries"], 3);
}

// ---------------------------------------------------------------------------
// Data source validation
// ---------------------------------------------------------------------------

class ValidateDataSourceTest : public ::testing::Test {
protected:
  std::vector<std::string> errors;
};

TEST_F(ValidateDataSourceTest, ValidMinimalConfig) {
  json config = {
      {"data", "iris.csv"},
      {"size", 100},
      {"lambda", 0.5},
      {"train_ratio", 0.7},
  };

  validate_training_config(config, errors);

  EXPECT_TRUE(errors.empty()) << errors[0];
}

TEST_F(ValidateDataSourceTest, ValidSimulateConfig) {
  json config = {
      {"simulate", "100x5x2"},
      {"size", 50},
      {"lambda", 0.5},
      {"train_ratio", 0.7},
      {"p_vars", 0.5},
  };

  validate_training_config(config, errors);

  EXPECT_TRUE(errors.empty()) << errors[0];
}

TEST_F(ValidateDataSourceTest, MissingDataSource) {
  json config = {
      {"size", 100},
      {"lambda", 0.5},
      {"train_ratio", 0.7},
  };

  validate_training_config(config, errors);

  EXPECT_FALSE(errors.empty());
  EXPECT_NE(errors[0].find("data source"), std::string::npos);
}

TEST_F(ValidateDataSourceTest, InvalidSimulateFormat) {
  json config = {
      {"simulate", "invalid"},
      {"size", 100},
      {"lambda", 0.5},
      {"train_ratio", 0.7},
  };

  validate_training_config(config, errors);

  bool has_sim_error = false;

  for (auto const& e : errors) {
    if (e.find("simulate") != std::string::npos) {
      has_sim_error = true;
    }
  }

  EXPECT_TRUE(has_sim_error);
}

TEST_F(ValidateDataSourceTest, SimulateGroupsMustBeGreaterThan1) {
  json config = {
      {"simulate", "100x5x1"},
      {"size", 100},
      {"lambda", 0.5},
      {"train_ratio", 0.7},
  };

  validate_training_config(config, errors);

  bool has_group_error = false;

  for (auto const& e : errors) {
    if (e.find("g must be") != std::string::npos) {
      has_group_error = true;
    }
  }

  EXPECT_TRUE(has_group_error);
}

// ---------------------------------------------------------------------------
// Integration: validate_training_config collects errors across all delegates
// ---------------------------------------------------------------------------

TEST(ValidateConfigIntegration, CollectsMultipleErrors) {
  json config = {
      {"size", -1},
      {"lambda", 2.0},
  };

  std::vector<std::string> errors;
  validate_training_config(config, errors);

  // Should have errors for: data source, size, lambda, train_ratio
  EXPECT_GE(errors.size(), 3U);
}

// ---------------------------------------------------------------------------
// Params::to_json
// ---------------------------------------------------------------------------

class ParamsToJsonTest : public ::testing::Test {};

TEST_F(ParamsToJsonTest, IncludesDataPath) {
  Params params;
  params.data_path = "iris.csv";

  auto config = params.to_json();

  EXPECT_EQ(config["data"], "iris.csv");
}

TEST_F(ParamsToJsonTest, IncludesSimulateFormat) {
  Params params;
  params.simulation.format = "100x5x2";

  auto config = params.to_json();

  EXPECT_EQ(config["simulate"], "100x5x2");
}

TEST_F(ParamsToJsonTest, IncludesModelSize) {
  Params params;

  auto config = params.to_json();

  EXPECT_EQ(config["size"], 100);
}

TEST_F(ParamsToJsonTest, IncludesLambdaWhenSet) {
  Params params;
  params.model.lambda = 0.3;

  auto config = params.to_json();

  EXPECT_FLOAT_EQ(config["lambda"].get<float>(), 0.3F);
}

TEST_F(ParamsToJsonTest, IncludesTrainRatio) {
  Params params;
  params.evaluate.train_ratio = 0.7F;

  auto config = params.to_json();

  EXPECT_FLOAT_EQ(config["train_ratio"].get<float>(), 0.7F);
}

TEST_F(ParamsToJsonTest, OmitsUnsetTrainRatio) {
  Params params;

  auto config = params.to_json();

  EXPECT_FALSE(config.contains("train_ratio"));
}

TEST_F(ParamsToJsonTest, OmitsUnsetOptionals) {
  Params params;

  auto config = params.to_json();

  // Unset optionals are omitted from config
  EXPECT_FALSE(config.contains("seed"));
  EXPECT_FALSE(config.contains("threads"));
  EXPECT_FALSE(config.contains("n_vars"));
  EXPECT_FALSE(config.contains("p_vars"));
}

TEST_F(ParamsToJsonTest, IncludesExplicitSeed) {
  Params params;
  params.model.seed = 123;

  auto config = params.to_json();

  EXPECT_EQ(config["seed"], 123);
}

TEST_F(ParamsToJsonTest, IncludesStrategyConfigs) {
  Params params;
  params.model.pp_config = {{"name", "pda"}, {"lambda", 0.3}};

  auto config = params.to_json();

  EXPECT_EQ(config["pp"]["name"], "pda");
}

TEST_F(ParamsToJsonTest, OmitsNullStrategyConfigs) {
  Params params;

  auto config = params.to_json();

  EXPECT_FALSE(config.contains("pp"));
  EXPECT_FALSE(config.contains("vars"));
  EXPECT_FALSE(config.contains("cutpoint"));
}

TEST_F(ParamsToJsonTest, DefaultConfigPassesValidation) {
  Params params;
  params.data_path = "iris.csv";
  params.resolve();
  params.evaluate.resolve_defaults();
  params.resolve_defaults(0);

  auto config = params.to_json();
  std::vector<std::string> errors;
  validate_training_config(config, errors);

  EXPECT_TRUE(errors.empty()) << errors[0];
}
