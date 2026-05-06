/**
 * @file ModelParams.test.cpp
 * @brief Unit tests for ModelParams helpers: parse_proportion (string and JSON).
 */
#include <gtest/gtest.h>

#include <algorithm>

#include "cli/ModelParams.hpp"
#include "utils/UserError.hpp"

using ppforest2::cli::parse_proportion;
using ppforest2::cli::strategy_string_to_json;
using json = nlohmann::json;

// ---------------------------------------------------------------------------
// parse_proportion — String inputs
// ---------------------------------------------------------------------------

TEST(ParseProportion, DecimalOne) {
  EXPECT_FLOAT_EQ(parse_proportion(std::string("1.0")), 1.0F);
}

TEST(ParseProportion, FractionOne) {
  EXPECT_FLOAT_EQ(parse_proportion(std::string("3/3")), 1.0F);
}

TEST(ParseProportion, Decimal) {
  EXPECT_FLOAT_EQ(parse_proportion(std::string("0.5")), 0.5F);
}

TEST(ParseProportion, Centesimal) {
  EXPECT_FLOAT_EQ(parse_proportion(std::string("0.01")), 0.01F);
}

TEST(ParseProportion, Fraction) {
  EXPECT_NEAR(parse_proportion(std::string("1/3")), 1.0F / 3.0F, 1e-6);
}

TEST(ParseProportion, DecimalZeroThrows) {
  EXPECT_THROW(parse_proportion(std::string("0.0")), ppforest2::UserError);
}

TEST(ParseProportion, DecimalAboveOneThrows) {
  EXPECT_THROW(parse_proportion(std::string("1.5")), ppforest2::UserError);
}

TEST(ParseProportion, NegativeDecimalThrows) {
  EXPECT_THROW(parse_proportion(std::string("-0.5")), ppforest2::UserError);
}

TEST(ParseProportion, ZeroIntegerThrows) {
  EXPECT_THROW(parse_proportion(std::string("0")), ppforest2::UserError);
}

TEST(ParseProportion, NegativeIntegerThrows) {
  EXPECT_THROW(parse_proportion(std::string("-1")), ppforest2::UserError);
}

TEST(ParseProportion, IntegerAboveOneThrows) {
  EXPECT_THROW(parse_proportion(std::string("10")), ppforest2::UserError);
}

TEST(ParseProportion, FractionAboveOneThrows) {
  EXPECT_THROW(parse_proportion(std::string("3/2")), ppforest2::UserError);
}

TEST(ParseProportion, FractionZeroDenominatorThrows) {
  EXPECT_THROW(parse_proportion(std::string("1/0")), ppforest2::UserError);
}

TEST(ParseProportion, FractionNegativeNumeratorThrows) {
  EXPECT_THROW(parse_proportion(std::string("-1/3")), ppforest2::UserError);
}

TEST(ParseProportion, FractionNegativeDenominatorThrows) {
  EXPECT_THROW(parse_proportion(std::string("1/-3")), ppforest2::UserError);
}

// ---------------------------------------------------------------------------
// parse_proportion — JSON inputs
// ---------------------------------------------------------------------------

TEST(ParseProportionJson, Float) {
  EXPECT_FLOAT_EQ(parse_proportion(json(0.3)), 0.3F);
}

TEST(ParseProportionJson, FloatOne) {
  EXPECT_FLOAT_EQ(parse_proportion(json(1.0)), 1.0F);
}

TEST(ParseProportionJson, StringFraction) {
  EXPECT_FLOAT_EQ(parse_proportion(json("2/5")), 0.4F);
}

TEST(ParseProportionJson, StringDecimal) {
  EXPECT_FLOAT_EQ(parse_proportion(json("0.7")), 0.7F);
}

TEST(ParseProportionJson, StringDecimalOne) {
  EXPECT_FLOAT_EQ(parse_proportion(json("1.0")), 1.0F);
}

TEST(ParseProportionJson, StringCentesimal) {
  EXPECT_FLOAT_EQ(parse_proportion(json("0.01")), 0.01F);
}

TEST(ParseProportionJson, FloatZeroThrows) {
  EXPECT_THROW(parse_proportion(json(0.0)), ppforest2::UserError);
}

TEST(ParseProportionJson, FloatAboveOneThrows) {
  EXPECT_THROW(parse_proportion(json(1.5)), ppforest2::UserError);
}

TEST(ParseProportionJson, NegativeFloatThrows) {
  EXPECT_THROW(parse_proportion(json(-0.5)), ppforest2::UserError);
}

TEST(ParseProportionJson, StringZeroThrows) {
  EXPECT_THROW(parse_proportion(json("0")), ppforest2::UserError);
}

TEST(ParseProportionJson, StringAboveOneThrows) {
  EXPECT_THROW(parse_proportion(json("10")), ppforest2::UserError);
}

TEST(ParseProportionJson, StringFloatZeroThrows) {
  EXPECT_THROW(parse_proportion(json("0.0")), ppforest2::UserError);
}

TEST(ParseProportionJson, StringFloatAboveOneThrows) {
  EXPECT_THROW(parse_proportion(json("1.5")), ppforest2::UserError);
}

TEST(ParseProportionJson, StringNegativeFloatThrows) {
  EXPECT_THROW(parse_proportion(json("-0.5")), ppforest2::UserError);
}

TEST(ParseProportionJson, StringNegativeIntegerThrows) {
  EXPECT_THROW(parse_proportion(json("-3")), ppforest2::UserError);
}

// ---------------------------------------------------------------------------
// strategy_string_to_json
// ---------------------------------------------------------------------------

TEST(StrategyStringToJson, NameOnly) {
  auto j = strategy_string_to_json("pda");
  EXPECT_EQ(j["name"], "pda");
  EXPECT_EQ(j.size(), 1);
}

TEST(StrategyStringToJson, NameWithOneParam) {
  auto j = strategy_string_to_json("pda:lambda=0.3");
  EXPECT_EQ(j["name"], "pda");
  EXPECT_DOUBLE_EQ(j["lambda"].get<double>(), 0.3);
  EXPECT_EQ(j.size(), 2);
}

TEST(StrategyStringToJson, NameWithMultipleParams) {
  auto j = strategy_string_to_json("custom:alpha=0.5,beta=2,gamma=3.14");
  EXPECT_EQ(j["name"], "custom");
  EXPECT_DOUBLE_EQ(j["alpha"].get<double>(), 0.5);
  EXPECT_EQ(j["beta"].get<int>(), 2);
  EXPECT_DOUBLE_EQ(j["gamma"].get<double>(), 3.14);
  EXPECT_EQ(j.size(), 4);
}

TEST(StrategyStringToJson, IntegerParam) {
  auto j = strategy_string_to_json("uniform:count=5");
  EXPECT_EQ(j["name"], "uniform");
  EXPECT_EQ(j["count"].get<int>(), 5);
  EXPECT_TRUE(j["count"].is_number_integer());
}

TEST(StrategyStringToJson, FloatParam) {
  auto j = strategy_string_to_json("uniform:proportion=0.5");
  EXPECT_EQ(j["name"], "uniform");
  EXPECT_DOUBLE_EQ(j["proportion"].get<double>(), 0.5);
  EXPECT_TRUE(j["proportion"].is_number_float());
}

TEST(StrategyStringToJson, StringParam) {
  auto j = strategy_string_to_json("custom:mode=fast");
  EXPECT_EQ(j["name"], "custom");
  EXPECT_EQ(j["mode"], "fast");
  EXPECT_TRUE(j["mode"].is_string());
}

TEST(StrategyStringToJson, NegativeInteger) {
  auto j = strategy_string_to_json("custom:offset=-3");
  EXPECT_EQ(j["offset"].get<int>(), -3);
  EXPECT_TRUE(j["offset"].is_number_integer());
}

TEST(StrategyStringToJson, NegativeFloat) {
  auto j = strategy_string_to_json("custom:weight=-0.5");
  EXPECT_DOUBLE_EQ(j["weight"].get<double>(), -0.5);
  EXPECT_TRUE(j["weight"].is_number_float());
}

TEST(StrategyStringToJson, ZeroInteger) {
  auto j = strategy_string_to_json("custom:count=0");
  EXPECT_EQ(j["count"].get<int>(), 0);
  EXPECT_TRUE(j["count"].is_number_integer());
}

TEST(StrategyStringToJson, MissingEqualsThrows) {
  // `pda:lambda` is not shorthand (the bare token is a string, not a
  // number), so this still goes through the key=value parser and fails
  // with the standard "expected key=value" error — guards against
  // accidentally interpreting `pda:lambda` as `{name: pda, lambda: "lambda"}`.
  EXPECT_THROW(strategy_string_to_json("pda:lambda"), std::runtime_error);
}

// ---------------------------------------------------------------------------
// strategy_string_to_json — positional shorthand
// ---------------------------------------------------------------------------

TEST(StrategyStringToJsonShorthand, MinSize) {
  auto j = strategy_string_to_json("min_size:5");
  EXPECT_EQ(j["name"], "min_size");
  EXPECT_EQ(j["min_size"].get<int>(), 5);
  EXPECT_EQ(j.size(), 2U);
}

TEST(StrategyStringToJsonShorthand, MinVariance) {
  // Primary param name is `threshold`, not the strategy name.
  auto j = strategy_string_to_json("min_variance:0.01");
  EXPECT_EQ(j["name"], "min_variance");
  EXPECT_DOUBLE_EQ(j["threshold"].get<double>(), 0.01);
  EXPECT_FALSE(j.contains("min_variance"));
}

TEST(StrategyStringToJsonShorthand, MaxDepth) {
  auto j = strategy_string_to_json("max_depth:8");
  EXPECT_EQ(j["name"], "max_depth");
  EXPECT_EQ(j["max_depth"].get<int>(), 8);
}

TEST(StrategyStringToJsonShorthand, PdaLambda) {
  auto j = strategy_string_to_json("pda:0.3");
  EXPECT_EQ(j["name"], "pda");
  EXPECT_DOUBLE_EQ(j["lambda"].get<double>(), 0.3);
}

TEST(StrategyStringToJsonShorthand, UniformCount) {
  auto j = strategy_string_to_json("uniform:3");
  EXPECT_EQ(j["name"], "uniform");
  EXPECT_EQ(j["count"].get<int>(), 3);
}

TEST(StrategyStringToJsonShorthand, ExplicitStillWorks) {
  // Explicit key=value must behave identically to pre-shorthand. The
  // shorthand path is skipped when the input contains `=`, so this
  // exercises the fallback.
  auto j = strategy_string_to_json("min_size:min_size=5");
  EXPECT_EQ(j["name"], "min_size");
  EXPECT_EQ(j["min_size"].get<int>(), 5);
}

TEST(StrategyStringToJsonShorthand, MultiParamStrategyBypassesShorthand) {
  // Strategies without a primary-param entry fall through: `custom:5`
  // is a missing-equals error, not silently a positional shorthand.
  EXPECT_THROW(strategy_string_to_json("custom:5"), std::runtime_error);
}

TEST(StrategyStringToJsonShorthand, CommaDisablesShorthand) {
  // A comma implies multiple params, so shorthand doesn't apply even if
  // the strategy has a primary-param entry. `min_size:5,extra=1` goes
  // through the key=value parser and fails on the `5` token.
  EXPECT_THROW(strategy_string_to_json("min_size:5,extra=1"), std::runtime_error);
}

TEST(StrategyStringToJson, MissingEqualsInSecondParamThrows) {
  EXPECT_THROW(strategy_string_to_json("pda:lambda=0.3,bad"), std::runtime_error);
}

TEST(StrategyStringToJson, EmptyNameThrows) {
  EXPECT_THROW(strategy_string_to_json(":key=val"), std::runtime_error);
}

TEST(StrategyStringToJson, EmptyValueParsesAsZero) {
  auto j = strategy_string_to_json("custom:key=");
  EXPECT_DOUBLE_EQ(j["key"].get<double>(), 0.0);
}

TEST(StrategyStringToJson, TrailingColonThrows) {
  EXPECT_THROW(strategy_string_to_json("pda:"), std::runtime_error);
}

// ---------------------------------------------------------------------------
// ModelParams::validate
// ---------------------------------------------------------------------------

class ValidateModelConfigTest : public ::testing::Test {
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

TEST_F(ValidateModelConfigTest, MissingSize) {
  json config = base_config();
  config.erase("size");

  ppforest2::cli::ModelParams::validate(config, errors);

  EXPECT_TRUE(has_error("size"));
}

TEST_F(ValidateModelConfigTest, NegativeSize) {
  json config    = base_config();
  config["size"] = -1;

  ppforest2::cli::ModelParams::validate(config, errors);

  EXPECT_TRUE(has_error("size"));
}

TEST_F(ValidateModelConfigTest, ZeroSizeIsValid) {
  json config    = base_config();
  config["size"] = 0;

  ppforest2::cli::ModelParams::validate(config, errors);

  EXPECT_FALSE(has_error("size"));
}

TEST_F(ValidateModelConfigTest, MissingLambdaWithoutPP) {
  json config = base_config();
  config.erase("lambda");

  ppforest2::cli::ModelParams::validate(config, errors);

  EXPECT_TRUE(has_error("lambda"));
}

TEST_F(ValidateModelConfigTest, InvalidLambdaRange) {
  json config      = base_config();
  config["lambda"] = 1.5;

  ppforest2::cli::ModelParams::validate(config, errors);

  EXPECT_TRUE(has_error("lambda"));
}

TEST_F(ValidateModelConfigTest, PPStrategySkipsLambdaRequirement) {
  json config = base_config();
  config.erase("lambda");
  config["pp"] = {{"name", "pda"}, {"lambda", 0.3}};

  ppforest2::cli::ModelParams::validate(config, errors);

  EXPECT_TRUE(errors.empty()) << errors[0];
}

TEST_F(ValidateModelConfigTest, InvalidPVars) {
  json config      = base_config();
  config["p_vars"] = 1.5;

  ppforest2::cli::ModelParams::validate(config, errors);

  EXPECT_TRUE(has_error("p_vars"));
}

TEST_F(ValidateModelConfigTest, NegativeNVars) {
  json config      = base_config();
  config["n_vars"] = -1;

  ppforest2::cli::ModelParams::validate(config, errors);

  EXPECT_TRUE(has_error("n_vars"));
}

TEST_F(ValidateModelConfigTest, ZeroNVars) {
  json config      = base_config();
  config["n_vars"] = 0;

  ppforest2::cli::ModelParams::validate(config, errors);

  EXPECT_TRUE(has_error("n_vars"));
}

TEST_F(ValidateModelConfigTest, ForestWithNVars) {
  json config      = base_config();
  config["n_vars"] = 3;

  ppforest2::cli::ModelParams::validate(config, errors);

  EXPECT_TRUE(errors.empty()) << errors[0];
}

TEST_F(ValidateModelConfigTest, ForestWithVarsStrategy) {
  json config    = base_config();
  config["vars"] = {{"name", "uniform"}, {"count", 3}};

  ppforest2::cli::ModelParams::validate(config, errors);

  EXPECT_TRUE(errors.empty()) << errors[0];
}

TEST_F(ValidateModelConfigTest, NegativeThreads) {
  json config       = base_config();
  config["threads"] = -5;

  ppforest2::cli::ModelParams::validate(config, errors);

  EXPECT_TRUE(has_error("threads"));
}

TEST_F(ValidateModelConfigTest, OmittedThreadsAreValid) {
  json config = base_config();

  ppforest2::cli::ModelParams::validate(config, errors);

  EXPECT_TRUE(errors.empty()) << errors[0];
}

TEST_F(ValidateModelConfigTest, TreeDoesNotRequireVars) {
  json config    = base_config();
  config["size"] = 0;

  ppforest2::cli::ModelParams::validate(config, errors);

  EXPECT_TRUE(errors.empty()) << errors[0];
}
