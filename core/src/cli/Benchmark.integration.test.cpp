/**
 * @file Benchmark.integration.test.cpp
 * @brief Integration tests for the benchmark subcommand.
 */
#include "cli/CLI.integration.hpp"

// ---------------------------------------------------------------------------
// Benchmark subcommand
// ---------------------------------------------------------------------------

static std::string const MINIMAL_SCENARIOS = R"({
  "name": "integration-test",
  "defaults": {
    "train_ratio": 0.7,
    "seed": 0,
    "lambda": 0.5,
    "p_vars": 0.5,
    "iterations": 1
  },
  "scenarios": [
    { "name": "tiny-forest", "n": 50, "p": 3, "g": 2, "size": 5 },
    { "name": "tiny-tree",   "n": 50, "p": 3, "g": 2, "size": 0 }
  ]
})";
namespace {
  TempFile write_scenarios() {
    TempFile f;
    {
      std::ofstream out(f.path());
      out << MINIMAL_SCENARIOS;
    }
    return f;
  }
}
/* Benchmark runs successfully with a scenarios file. */
TEST(CLIBenchmark, BenchmarkRunsSuccessfully) {
  auto scenarios = write_scenarios();
  auto result    = run_ppforest2("-q --no-color benchmark -s " + scenarios.path());
  EXPECT_EQ(result.exit_code, 0);
}

/* Benchmark without -s must fail. */
TEST(CLIBenchmark, BenchmarkNoScenariosFails) {
  auto result = run_ppforest2("-q --no-color benchmark");
  EXPECT_NE(result.exit_code, 0);
}

/* Benchmark with invalid scenarios file must fail. */
TEST(CLIBenchmark, BenchmarkInvalidScenariosFails) {
  TempFile const bad;
  {
    std::ofstream out(bad.path());
    out << "not valid json";
  }
  auto result = run_ppforest2("-q --no-color benchmark -s " + bad.path());
  EXPECT_NE(result.exit_code, 0);
}

/* A wrong-typed scenario field fails validation with a per-scenario error. */
TEST(CLIBenchmark, BenchmarkWrongTypedScenarioFieldFails) {
  TempFile const bad;
  {
    std::ofstream out(bad.path());
    out << R"({"scenarios": [{ "name": "typed", "n": 50, "p": 3, "g": 2, "size": "10", "seed": 0 }]})";
  }

  auto result = run_ppforest2("-q --no-color benchmark -s " + bad.path());
  EXPECT_NE(result.exit_code, 0);
  EXPECT_NE(result.stderr_output.find("wrong-typed field"), std::string::npos);
}

/* Benchmark -o produces valid JSON results. */
TEST(CLIBenchmark, BenchmarkJsonOutput) {
  auto scenarios = write_scenarios();
  TempFile const output;
  output.clear();

  auto result = run_ppforest2("-q --no-color benchmark -s " + scenarios.path() + " -o " + output.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(output.read());
  EXPECT_TRUE(j.contains("suite_name"));
  EXPECT_TRUE(j.contains("timestamp"));
  EXPECT_TRUE(j.contains("total_time_ms"));
  EXPECT_TRUE(j.contains("results"));
  EXPECT_EQ(j["results"].size(), 2U);

  for (auto const& r : j["results"]) {
    EXPECT_TRUE(r.contains("name"));
    EXPECT_TRUE(r.contains("n"));
    EXPECT_TRUE(r.contains("p"));
    EXPECT_TRUE(r.contains("g"));
    EXPECT_TRUE(r.contains("size"));
    EXPECT_TRUE(r.contains("mean_time_ms"));
    EXPECT_TRUE(r.contains("mean_test_error"));
  }
}

/* Benchmark -o with .csv extension produces valid CSV with header and data rows. */
TEST(CLIBenchmark, BenchmarkCsvOutput) {
  auto scenarios = write_scenarios();
  TempFile const csv_out(".csv");
  csv_out.clear();

  auto result = run_ppforest2("-q --no-color benchmark -s " + scenarios.path() + " -o " + csv_out.path());
  EXPECT_EQ(result.exit_code, 0);

  std::ifstream in(csv_out.path());
  std::string header;
  std::getline(in, header);
  EXPECT_NE(header.find("scenario"), std::string::npos);
  EXPECT_NE(header.find("mean_time_ms"), std::string::npos);

  // Two data rows (one per scenario)
  std::string line;
  int data_rows = 0;

  while (std::getline(in, line)) {
    if (!line.empty()) {
      data_rows++;
    }
  }

  EXPECT_EQ(data_rows, 2);
}

/* Benchmark -o can be repeated to produce both JSON and CSV simultaneously. */
TEST(CLIBenchmark, BenchmarkBothOutputFormats) {
  auto scenarios = write_scenarios();
  TempFile const json_out;
  TempFile const csv_out(".csv");
  json_out.clear();
  csv_out.clear();

  auto result = run_ppforest2(
      "-q --no-color benchmark -s " + scenarios.path() + " -o " + json_out.path() + " -o " + csv_out.path()
  );
  EXPECT_EQ(result.exit_code, 0);

  // JSON output is valid
  auto j = json::parse(json_out.read());
  EXPECT_EQ(j["results"].size(), 2U);

  // CSV output has header + 2 data rows
  std::ifstream in(csv_out.path());
  std::string line;
  int lines = 0;

  while (std::getline(in, line)) {
    if (!line.empty()) {
      lines++;
    }
  }

  EXPECT_EQ(lines, 3); // header + 2 data rows
}

/* Benchmark -b compares against a baseline without error. */
TEST(CLIBenchmark, BenchmarkBaselineComparison) {
  auto scenarios = write_scenarios();

  // First run: produce baseline
  TempFile const baseline;
  baseline.clear();
  auto run1 = run_ppforest2("-q --no-color benchmark -s " + scenarios.path() + " -o " + baseline.path());
  ASSERT_EQ(run1.exit_code, 0) << "stderr: " << run1.stderr_output;

  // Second run: compare against baseline
  auto run2 = run_ppforest2("-q --no-color benchmark -s " + scenarios.path() + " -b " + baseline.path());
  EXPECT_EQ(run2.exit_code, 0) << "stderr: " << run2.stderr_output;
}

/* Benchmark -i overrides iteration count. */
TEST(CLIBenchmark, BenchmarkIterationOverride) {
  auto scenarios = write_scenarios();
  TempFile const output;
  output.clear();

  auto result = run_ppforest2("-q --no-color benchmark -s " + scenarios.path() + " -i 2 -o " + output.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(output.read());
  for (auto const& r : j["results"]) {
    EXPECT_EQ(r["runs"].get<int>(), 2);
  }
}

/* Benchmark with missing scenarios file must fail. */
TEST(CLIBenchmark, BenchmarkMissingFileFails) {
  auto result = run_ppforest2("-q --no-color benchmark -s /nonexistent/path.json");
  EXPECT_NE(result.exit_code, 0);
}

// ---------------------------------------------------------------------------
// Benchmark with explicit strategy config
// ---------------------------------------------------------------------------

/* Benchmark scenarios with explicit strategy objects run successfully. */
TEST(CLIBenchmark, BenchmarkWithStrategyConfig) {
  static std::string const STRATEGY_SCENARIOS = R"({
    "name": "strategy-test",
    "defaults": {
      "train_ratio": 0.7,
      "seed": 0,
      "p_vars": 0.5,
      "iterations": 1,
      "pp": { "name": "pda", "lambda": 0.3 },
      "cutpoint": { "name": "mean_of_means" }
    },
    "scenarios": [
      { "name": "strat-forest", "n": 50, "p": 3, "g": 2, "size": 5, "p_vars": 0.67 },
      { "name": "strat-tree",   "n": 50, "p": 3, "g": 2, "size": 0 }
    ]
  })";

  TempFile const scenarios;
  {
    std::ofstream out(scenarios.path());
    out << STRATEGY_SCENARIOS;
  }

  TempFile const output;
  output.clear();
  auto result = run_ppforest2("-q --no-color benchmark -s " + scenarios.path() + " -o " + output.path());
  EXPECT_EQ(result.exit_code, 0);

  auto j = json::parse(output.read());
  EXPECT_EQ(j["results"].size(), 2U);
}
