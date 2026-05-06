/**
 * @file Benchmark.hpp
 * @brief Benchmark scenario types, JSON parsing, and subprocess-based execution.
 *
 * Each benchmark scenario runs as a separate `ppforest2 evaluate` process,
 * giving accurate per-scenario peak RSS measurements.
 *
 * Scenarios are represented as plain JSON objects throughout — no
 * intermediate typed struct.  Default merging, validation, and config
 * extraction all operate on JSON directly.
 */
#pragma once

#include "cli/CLIOptions.hpp"
#include "io/EvaluateResult.hpp"

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace CLI {
  class App;
}

namespace ppforest2::cli {
  /** @brief Register benchmark subcommand options on @p app. */
  void setup_benchmark(CLI::App& app, Params& params);

  /**
   * @brief Result of running a single benchmark scenario.
   *
   * Inherits evaluate metrics (n, p, g, timing, errors, RSS) from
   * EvaluateResult and adds scenario-specific metadata for reporting.
   */
  struct ScenarioResult : io::EvaluateResult {
    ScenarioResult() = default;

    ScenarioResult(std::string name, double scenario_time_ms, nlohmann::json const& j)
        : io::EvaluateResult(j)
        , name(std::move(name))
        , scenario_time_ms(scenario_time_ms) {}

    std::string name;

    // Scenario wall-clock time (including warmup + process overhead)
    double scenario_time_ms = 0;
  };

  /**
   * @brief A suite of scenarios with shared defaults.
   *
   * Each scenario is a plain JSON object containing all fields
   * (after default merging).
   */
  struct BenchmarkSuite {
    explicit BenchmarkSuite(std::string name = "ppforest2 benchmark")
        : name(std::move(name)) {}

    std::string name;
    std::vector<nlohmann::json> scenarios;
  };

  /**
   * @brief Aggregated results for an entire suite run.
   */
  struct SuiteResult {
    SuiteResult() = default;
    explicit SuiteResult(nlohmann::json const& j);

    std::string suite_name;
    std::string timestamp;
    std::vector<ScenarioResult> results;
    double total_time_ms = 0;

    nlohmann::json to_json() const;
    std::string to_csv() const;
  };

  /**
   * @brief Parse a BenchmarkSuite from a JSON file path.
   * @throws UserError on parse or validation errors.
   */
  BenchmarkSuite parse_suite(std::string const& path);

  /**
   * @brief Parse a BenchmarkSuite from a JSON object.
   * @throws UserError on validation errors.
   */
  BenchmarkSuite parse_suite(nlohmann::json const& j);

  /**
   * @brief Run all scenarios in a suite via subprocess invocations.
   *
   * For each scenario, spawns `ppforest2 evaluate` as a child process with
   * the appropriate flags, reads its JSON output, and collects results.
   * Progress is printed to @p out.
   *
   * @param suite       The benchmark suite to run.
   * @param binary_path Path to the ppforest2 binary (typically argv[0]).
   * @param out         Output context for progress reporting.
   * @return Aggregated suite results.
   */
  SuiteResult run_suite(BenchmarkSuite const& suite, std::string const& binary_path, io::Output& out);

  /**
   * @brief Run a single scenario as a subprocess.
   */
  ScenarioResult run_scenario(nlohmann::json const& scenario, std::string const& binary_path);

  /**
   * @brief Run the benchmark subcommand.
   * @param params      CLI options.
   * @param binary_path Path to the ppforest2 binary (typically argv[0]).
   * @return Exit code.
   */
  int run_benchmark(Params const& params, std::string const& binary_path);
}
