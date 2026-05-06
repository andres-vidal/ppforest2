/**
 * @file Benchmark.test.cpp
 * @brief Tests for benchmark scenario parsing, validation, and convergence.
 */
#include <gtest/gtest.h>
#include "cli/Benchmark.hpp"
#include "utils/UserError.hpp"

using namespace ppforest2::cli;
using json = nlohmann::json;

class BenchmarkParsingTest : public ::testing::Test {};

TEST_F(BenchmarkParsingTest, ParseMinimalScenario) {
  json const j = {
      {"scenarios",
       {{{"name", "test"},
         {"n", 100},
         {"p", 5},
         {"g", 2},
         {"size", 50},
         {"seed", 0},
         {"p_vars", 0.5},
         {"lambda", 0.3},
         {"train_ratio", 0.7}}}}
  };

  auto suite = parse_suite(j);

  ASSERT_EQ(suite.scenarios.size(), 1U);
  EXPECT_EQ(suite.scenarios[0]["name"], "test");
  EXPECT_EQ(suite.scenarios[0]["n"], 100);
  EXPECT_EQ(suite.scenarios[0]["p"], 5);
  EXPECT_EQ(suite.scenarios[0]["g"], 2);
  EXPECT_EQ(suite.scenarios[0]["size"], 50);
  EXPECT_FLOAT_EQ(suite.scenarios[0]["p_vars"].get<float>(), 0.5F);
  EXPECT_FLOAT_EQ(suite.scenarios[0]["lambda"].get<float>(), 0.3F);
  EXPECT_FLOAT_EQ(suite.scenarios[0]["train_ratio"].get<float>(), 0.7F);
}

TEST_F(BenchmarkParsingTest, IncompleteScenarioThrows) {
  json const j = {{"scenarios", {{{"name", "test"}, {"n", 100}, {"p", 5}, {"g", 2}}}}};

  EXPECT_THROW(parse_suite(j), ppforest2::UserError);
}

TEST_F(BenchmarkParsingTest, ParseWithDefaults) {
  json const j = {
      {"defaults", {{"size", 50}, {"lambda", 0.3}, {"seed", 123}, {"warmup", 3}, {"p_vars", 0.5}, {"train_ratio", 0.7}}
      },
      {"scenarios",
       {{{"name", "a"}, {"n", 200}, {"p", 10}, {"g", 3}},
        {{"name", "b"}, {"n", 500}, {"p", 20}, {"g", 4}, {"size", 200}}}}
  };

  auto suite = parse_suite(j);

  ASSERT_EQ(suite.scenarios.size(), 2U);

  // Scenario a inherits all defaults
  EXPECT_EQ(suite.scenarios[0]["size"], 50);
  EXPECT_FLOAT_EQ(suite.scenarios[0]["lambda"].get<float>(), 0.3F);
  EXPECT_EQ(suite.scenarios[0]["seed"], 123);
  EXPECT_EQ(suite.scenarios[0]["warmup"], 3);

  // Scenario b overrides size
  EXPECT_EQ(suite.scenarios[1]["size"], 200);
  EXPECT_FLOAT_EQ(suite.scenarios[1]["lambda"].get<float>(), 0.3F);
  EXPECT_EQ(suite.scenarios[1]["seed"], 123);
}

TEST_F(BenchmarkParsingTest, ParseWithConvergence) {
  json const j = {
      {"defaults",
       {{"size", 50},
        {"lambda", 0.5},
        {"seed", 0},
        {"p_vars", 0.5},
        {"train_ratio", 0.7},
        {"convergence", {{"cv", 0.03}, {"max", 100}}}}},
      {"scenarios", {{{"name", "a"}, {"n", 100}, {"p", 5}, {"g", 2}}}}
  };

  auto suite = parse_suite(j);

  EXPECT_FLOAT_EQ(suite.scenarios[0]["convergence"]["cv"].get<float>(), 0.03F);
  EXPECT_EQ(suite.scenarios[0]["convergence"]["max"], 100);
}

TEST_F(BenchmarkParsingTest, ConvergenceDefaultsAppliedWhenOmitted) {
  json const j = {
      {"defaults", {{"size", 50}, {"seed", 0}, {"lambda", 0.5}, {"p_vars", 0.5}, {"train_ratio", 0.7}}},
      {"scenarios", {{{"name", "a"}, {"n", 100}, {"p", 5}, {"g", 2}}}}
  };

  auto suite = parse_suite(j);

  auto const& conv = suite.scenarios[0]["convergence"];
  EXPECT_FLOAT_EQ(conv["cv"].get<float>(), 0.05F);
  EXPECT_EQ(conv["window"], 3);
  EXPECT_EQ(conv["min"], 10);
  EXPECT_EQ(conv["max"], 200);
}

TEST_F(BenchmarkParsingTest, ConvergenceDefaultsMergedWithPartialOverride) {
  json const j = {
      {"defaults", {{"size", 50}, {"seed", 0}, {"lambda", 0.5}, {"p_vars", 0.5}, {"train_ratio", 0.7}}},
      {"scenarios", {{{"name", "a"}, {"n", 100}, {"p", 5}, {"g", 2}, {"convergence", {{"cv", 0.1}, {"max", 50}}}}}}
  };

  auto suite = parse_suite(j);

  auto const& conv = suite.scenarios[0]["convergence"];
  EXPECT_FLOAT_EQ(conv["cv"].get<float>(), 0.1F);
  EXPECT_EQ(conv["max"], 50);
  // Defaults preserved for unspecified keys
  EXPECT_EQ(conv["window"], 3);
  EXPECT_EQ(conv["min"], 10);
}

TEST_F(BenchmarkParsingTest, FixedIterationsOverrideConvergence) {
  json const j = {
      {"defaults",
       {{"size", 50},
        {"seed", 0},
        {"lambda", 0.5},
        {"p_vars", 0.5},
        {"train_ratio", 0.7},
        {"convergence", {{"max", 100}}}}},
      {"scenarios", {{{"name", "fixed"}, {"n", 100}, {"p", 5}, {"g", 2}, {"iterations", 5}}}}
  };

  auto suite = parse_suite(j);

  EXPECT_EQ(suite.scenarios[0]["iterations"], 5);
}

TEST_F(BenchmarkParsingTest, ParseSuiteName) {
  json const j = {
      {"name", "my benchmarks"},
      {"defaults", {{"size", 50}, {"seed", 0}, {"lambda", 0.5}, {"p_vars", 0.5}, {"train_ratio", 0.7}}},
      {"scenarios", {{{"name", "a"}, {"n", 100}, {"p", 5}, {"g", 2}}}}
  };

  auto suite = parse_suite(j);

  EXPECT_EQ(suite.name, "my benchmarks");
}

TEST_F(BenchmarkParsingTest, MissingScenariosArrayThrows) {
  json const j = {{"defaults", {}}};

  EXPECT_THROW(parse_suite(j), std::runtime_error);
}

TEST_F(BenchmarkParsingTest, EmptyScenariosArrayThrows) {
  json const j = {{"scenarios", json::array()}};

  EXPECT_THROW(parse_suite(j), std::runtime_error);
}

TEST_F(BenchmarkParsingTest, InvalidScenarioThrows) {
  // g must be > 1
  json const j = {
      {"defaults", {{"size", 50}, {"lambda", 0.5}, {"p_vars", 0.5}, {"train_ratio", 0.7}}},
      {"scenarios", {{{"name", "bad"}, {"n", 100}, {"p", 5}, {"g", 1}}}}
  };

  EXPECT_THROW(parse_suite(j), ppforest2::UserError);
}

TEST_F(BenchmarkParsingTest, MissingNameThrows) {
  json const j = {
      {"defaults", {{"size", 50}, {"lambda", 0.5}, {"p_vars", 0.5}, {"train_ratio", 0.7}}},
      {"scenarios", {{{"n", 100}, {"p", 5}, {"g", 2}}}}
  };

  EXPECT_THROW(parse_suite(j), ppforest2::UserError);
}

TEST_F(BenchmarkParsingTest, NegativeNThrows) {
  json const j = {
      {"defaults", {{"size", 50}, {"lambda", 0.5}, {"p_vars", 0.5}, {"train_ratio", 0.7}}},
      {"scenarios", {{{"name", "bad"}, {"n", -1}, {"p", 5}, {"g", 2}}}}
  };

  EXPECT_THROW(parse_suite(j), ppforest2::UserError);
}

TEST_F(BenchmarkParsingTest, MissingSeedThrows) {
  json const j = {
      {"defaults", {{"size", 50}, {"lambda", 0.5}, {"p_vars", 0.5}, {"train_ratio", 0.7}}},
      {"scenarios", {{{"name", "no-seed"}, {"n", 100}, {"p", 5}, {"g", 2}}}}
  };

  EXPECT_THROW(parse_suite(j), ppforest2::UserError);
}

// Parameterised across the two bench scenario files (classification +
// regression). Both must parse cleanly and both must satisfy the same
// structural invariants — shared validation means the two files don't
// drift into incompatible shapes over time.
class DefaultScenariosFileTest : public BenchmarkParsingTest, public ::testing::WithParamInterface<std::string> {};

TEST_P(DefaultScenariosFileTest, Parses) {
  auto suite = parse_suite(GetParam());

  EXPECT_GT(suite.scenarios.size(), 0U);

  for (auto const& s : suite.scenarios) {
    EXPECT_TRUE(s.contains("name"));
    EXPECT_FALSE(s["name"].get<std::string>().empty());

    if (!s.contains("data") || s["data"].get<std::string>().empty()) {
      // Simulation scenarios must have valid dimensions
      EXPECT_GT(s["n"].get<int>(), 0);
      EXPECT_GT(s["p"].get<int>(), 0);

      bool is_regression = s.contains("mode") && s["mode"].get<std::string>() == "regression";
      if (!is_regression) {
        // Classification simulation requires `g` (number of groups).
        EXPECT_GT(s["g"].get<int>(), 1);
      }
    } else {
      // Data scenarios must have a non-empty path
      EXPECT_FALSE(s["data"].get<std::string>().empty());
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    All,
    DefaultScenariosFileTest,
    ::testing::Values(std::string(PPFOREST2_BENCH_SCENARIOS_CLS), std::string(PPFOREST2_BENCH_SCENARIOS_REG))
);

class SuiteResultTest : public ::testing::Test {};

TEST_F(SuiteResultTest, ToJsonRoundtrip) {
  SuiteResult result;
  result.suite_name    = "test suite";
  result.timestamp     = "2026-01-01T00:00:00";
  result.total_time_ms = 1234.5;

  ScenarioResult sr;
  sr.name           = "scenario1";
  sr.n              = 100;
  sr.p              = 5;
  sr.g              = 2;
  sr.size           = 50;
  sr.p_vars         = 0.5F;
  sr.runs           = 10;
  sr.mean_time_ms   = 12.3;
  sr.std_time_ms    = 1.2;
  sr.mean_tr_error  = 0.01;
  sr.mean_te_error  = 0.05;
  sr.peak_rss_bytes = 1024UL * 1024UL;
  sr.peak_rss_mb    = 1.0;

  result.results.push_back(sr);

  auto j = result.to_json();

  EXPECT_EQ(j["suite_name"], "test suite");
  EXPECT_EQ(j["timestamp"], "2026-01-01T00:00:00");
  EXPECT_DOUBLE_EQ(j["total_time_ms"], 1234.5);

  ASSERT_EQ(j["results"].size(), 1U);
  EXPECT_EQ(j["results"][0]["name"], "scenario1");
  EXPECT_EQ(j["results"][0]["n"], 100);
  EXPECT_EQ(j["results"][0]["size"], 50);
  EXPECT_DOUBLE_EQ(j["results"][0]["mean_time_ms"], 12.3);
  EXPECT_EQ(j["results"][0]["peak_rss_bytes"], 1024 * 1024);
}

TEST_F(SuiteResultTest, ToJsonOmitsPeakRSSWhenUnset) {
  SuiteResult result;
  result.suite_name = "test";

  ScenarioResult sr;
  sr.name = "no-rss";
  // peak_rss_bytes and peak_rss_mb are std::nullopt by default

  result.results.push_back(sr);

  auto j = result.to_json();

  EXPECT_FALSE(j["results"][0].contains("peak_rss_bytes"));
  EXPECT_FALSE(j["results"][0].contains("peak_rss_mb"));
}

TEST_F(BenchmarkParsingTest, ParseDataScenario) {
  json const j = {
      {"defaults", {{"seed", 0}, {"lambda", 0.5}, {"p_vars", 0.5}, {"train_ratio", 0.7}}},
      {"scenarios", {{{"name", "csv-test"}, {"data", "data/classification/iris.csv"}, {"size", 50}}}}
  };

  auto suite = parse_suite(j);

  ASSERT_EQ(suite.scenarios.size(), 1U);
  EXPECT_EQ(suite.scenarios[0]["name"], "csv-test");
  EXPECT_EQ(suite.scenarios[0]["data"], "data/classification/iris.csv");
  EXPECT_EQ(suite.scenarios[0]["size"], 50);
}

TEST_F(BenchmarkParsingTest, DataScenarioSkipsNPGValidation) {
  // Data scenarios don't require explicit n/p/g
  json const j = {
      {"defaults", {{"size", 50}, {"seed", 0}, {"lambda", 0.5}, {"p_vars", 0.5}, {"train_ratio", 0.7}}},
      {"scenarios", {{{"name", "csv-test"}, {"data", "some/file.csv"}}}}
  };

  EXPECT_NO_THROW(parse_suite(j));
}

TEST_F(BenchmarkParsingTest, DataScenarioToJsonIncludesDataField) {
  SuiteResult result;
  result.suite_name = "test";

  ScenarioResult sr;
  sr.name      = "csv-test";
  sr.data_path = "data/classification/iris.csv";
  sr.n         = 150;
  sr.p         = 4;
  sr.g         = 3;

  result.results.push_back(sr);

  auto j = result.to_json();

  EXPECT_EQ(j["results"][0]["data"], "data/classification/iris.csv");
  EXPECT_EQ(j["results"][0]["n"], 150);
}

TEST_F(BenchmarkParsingTest, SimulationScenarioToJsonOmitsDataField) {
  SuiteResult result;
  result.suite_name = "test";

  ScenarioResult sr;
  sr.name = "sim-test";
  sr.n    = 100;
  sr.p    = 5;
  sr.g    = 2;

  result.results.push_back(sr);

  auto j = result.to_json();

  EXPECT_FALSE(j["results"][0].contains("data"));
}

TEST_F(BenchmarkParsingTest, ScenarioNVarsAsIntegerCount) {
  json const j = {
      {"defaults", {{"size", 50}, {"seed", 0}, {"lambda", 0.5}, {"train_ratio", 0.7}}},
      {"scenarios", {{{"name", "int-vars"}, {"n", 100}, {"p", 10}, {"g", 2}, {"n_vars", 3}}}}
  };

  auto suite = parse_suite(j);

  EXPECT_EQ(suite.scenarios[0]["n_vars"], 3);
}

TEST_F(BenchmarkParsingTest, NVarsRoundtripsInJson) {
  SuiteResult result;
  result.suite_name = "test";

  ScenarioResult sr;
  sr.name   = "with-nvars";
  sr.n_vars = 3;

  result.results.push_back(sr);

  auto j = result.to_json();

  EXPECT_EQ(j["results"][0]["n_vars"], 3);
}

TEST_F(BenchmarkParsingTest, NVarsOmittedFromJsonWhenNotSet) {
  SuiteResult result;
  result.suite_name = "test";

  ScenarioResult sr;
  sr.name = "no-nvars";

  result.results.push_back(sr);

  auto j = result.to_json();

  EXPECT_FALSE(j["results"][0].contains("n_vars"));
}

TEST_F(BenchmarkParsingTest, ScenarioPVarsAsFractionPassedThrough) {
  json const j = {
      {"defaults", {{"size", 50}, {"seed", 0}, {"lambda", 0.5}, {"train_ratio", 0.7}}},
      {"scenarios", {{{"name", "frac-vars"}, {"n", 100}, {"p", 10}, {"g", 2}, {"p_vars", "1/3"}}}}
  };

  auto suite = parse_suite(j);

  // Fraction strings are passed through to evaluate, which handles parsing
  EXPECT_EQ(suite.scenarios[0]["p_vars"], "1/3");
}
