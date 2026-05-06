/**
 * @file Benchmark.cpp
 * @brief Benchmark scenario parsing, validation, and subprocess execution.
 */
#include "cli/Benchmark.hpp"
#include "cli/BenchmarkReport.hpp"
#include "cli/CLIOptions.hpp"
#include "cli/Evaluate.hpp"
#include "cli/Validation.hpp"

#include <CLI/CLI.hpp>
#include "io/IO.hpp"
#include "io/TempFile.hpp"
#include "io/Timing.hpp"
#include "io/Color.hpp"
#include "io/Output.hpp"
#include "utils/UserError.hpp"

#include <fmt/format.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <optional>

#ifdef _WIN32
#include <process.h>
#else
#include <sys/wait.h>
#endif

namespace ppforest2::cli {
  void setup_benchmark(CLI::App& app, Params& params) {
    auto* sub = app.add_subcommand("benchmark", "Run performance benchmarks across scenarios");
    sub->add_option("-s,--scenarios", params.benchmark.scenarios_path, "JSON scenarios file");
    sub->add_option("-b,--baseline", params.benchmark.baseline_path, "Baseline results JSON for comparison");
    sub->add_option("-o,--output", params.benchmark.outputs, "Save results (.json or .csv, repeatable)");
    add_evaluate_options(sub, params.evaluate);
    sub->add_option("--format", params.benchmark.format, "Output format (table, markdown)");

    sub->get_option("--scenarios")->required()->check(CLI::ExistingFile);
    sub->get_option("--baseline")->check(CLI::ExistingFile);
    sub->get_option("--format")->check(CLI::IsMember({"table", "markdown"}));

    sub->callback([&]() { params.subcommand = Subcommand::benchmark; });
  }

  namespace {
    std::string format_errors(std::string const& name, std::vector<std::string> const& errors) {
      std::string result = fmt::format("Scenario '{}':\n", name);
      for (auto const& e : errors) {
        result += fmt::format("  - {}\n", e);
      }
      return result;
    }

    nlohmann::json const scenario_defaults = {{"convergence", {{"cv", 0.05F}, {"window", 3}, {"min", 10}, {"max", 200}}}
    };

    /**
     * @brief Build the JSON config for `ppforest2 evaluate --config`.
     *
     * Copies the scenario JSON and synthesizes a --simulate string
     * from n/p/g fields when no data path is provided.
     */
    nlohmann::json build_evaluate_config(nlohmann::json const& s) {
      nlohmann::json config = s;

      // Data source: convert n/p/g to --simulate if no data path
      if (!s.contains("data") || s["data"].get<std::string>().empty()) {
        config["simulate"] = fmt::format("{}x{}x{}", s.value("n", 1000), s.value("p", 10), s.value("g", 3));
      }

      return config;
    }

    /**
     * @brief Validate a benchmark scenario JSON object.
     *
     * Builds the evaluate config and delegates to central validation.
     * Adds scenario-specific checks (name, vars for forests).
     */
    std::vector<std::string> validate_scenario(nlohmann::json const& s) {
      std::vector<std::string> errors;

      check(s.contains("seed"), "seed is required for reproducibility", errors);

      auto config    = build_evaluate_config(s);
      bool is_forest = s.contains("size") && s["size"].get<int>() > 0;
      validate_training_config(config, errors);

      if (is_forest && !s.contains("vars")) {
        bool has_p_vars = s.contains("p_vars");
        bool has_n_vars = s.contains("n_vars") && s["n_vars"].get<int>() > 0;
        check(has_p_vars || has_n_vars, "p_vars or n_vars is required for forests", errors);
      }

      return errors;
    }
  }

  BenchmarkSuite parse_suite(std::string const& path) {
    std::ifstream file(path);

    user_error(file.is_open(), fmt::format("Failed to open scenarios file: {}", path));

    nlohmann::json j;

    try {
      j = nlohmann::json::parse(file);
    } catch (nlohmann::json::parse_error const& e) {
      throw ppforest2::UserError(fmt::format("Invalid JSON in scenarios file: {}", e.what()));
    }

    return parse_suite(j);
  }


  BenchmarkSuite parse_suite(nlohmann::json const& j) {
    BenchmarkSuite suite(j.value("name", std::string("ppforest2 benchmark")));

    nlohmann::json defaults = scenario_defaults;
    defaults.merge_patch(j.value("defaults", nlohmann::json::object()));

    user_error(j.contains("scenarios") && j["scenarios"].is_array(), "Scenarios file must contain a 'scenarios' array");
    user_error(!j["scenarios"].empty(), "Scenarios array must not be empty");

    std::string all_errors;

    int index = 0;

    for (auto const& scenario_json : j["scenarios"]) {
      nlohmann::json s = defaults;
      s.merge_patch(scenario_json);

      s["name"] = s.value("name", fmt::format("#{}", index + 1));

      auto errors = validate_scenario(s);
      if (!errors.empty()) {
        all_errors += format_errors(s["name"].get<std::string>(), errors);
      }

      suite.scenarios.push_back(std::move(s));
      ++index;
    }

    user_error(all_errors.empty(), all_errors);

    return suite;
  }

  SuiteResult::SuiteResult(nlohmann::json const& j)
      : suite_name(j.value("suite_name", ""))
      , timestamp(j.value("timestamp", ""))
      , total_time_ms(j.value("total_time_ms", 0.0)) {
    if (j.contains("results") && j["results"].is_array()) {
      for (auto const& r : j["results"]) {
        results.emplace_back(r.value("name", ""), r.value("scenario_time_ms", 0.0), r);
      }
    }
  }


  nlohmann::json SuiteResult::to_json() const {
    nlohmann::json j;
    j["suite_name"]    = suite_name;
    j["timestamp"]     = timestamp;
    j["total_time_ms"] = total_time_ms;

    nlohmann::json results_array = nlohmann::json::array();

    for (auto const& r : results) {
      nlohmann::json rj = r.io::EvaluateResult::to_json();

      rj["name"]             = r.name;
      rj["scenario_time_ms"] = r.scenario_time_ms;

      results_array.push_back(rj);
    }

    j["results"] = results_array;
    return j;
  }

  std::string SuiteResult::to_csv() const {
    std::string csv;

    csv += "scenario,n,p,g,trees,vars,train_ratio,runs,mean_time_ms,std_time_ms,"
           "mean_train_error,mean_test_error,peak_rss_mb,scenario_time_ms\n";

    for (auto const& r : results) {
      csv += fmt::format(
          "{},{},{},{},{},{:.2f},{:.2f},{},{:.2f},{:.2f},{:.4f},{:.4f},{:.1f},{:.0f}\n",
          r.name,
          r.n,
          r.p,
          r.g,
          r.size,
          r.p_vars.value_or(0),
          r.train_ratio,
          r.runs,
          r.mean_time_ms,
          r.std_time_ms,
          r.mean_tr_error,
          r.mean_te_error,
          r.peak_rss_mb.value_or(-1.0),
          r.scenario_time_ms
      );
    }

    return csv;
  }

  /**
   * @brief Run a command and return its exit code.
   *
   * Uses popen/pclose instead of std::system() to get reliable
   * exit codes on all platforms.  On Windows, std::system() wraps
   * via cmd.exe which can return unexpected exit codes.
   */
  int run_command(std::string const& cmd) {
    // clang-format off
    #ifdef _WIN32
    std::string full = "\"" + cmd + "\" 2>NUL";
    FILE* pipe       = _popen(full.c_str(), "r");
    #else
    std::string const full = cmd + " 2>/dev/null";
    FILE* pipe             = popen(full.c_str(), "r");
    #endif
    // clang-format on
    if (!pipe) {
      return -1;
    }

    // Drain the pipe (stdout is empty due to -q, but must be consumed)
    char buffer[4096];

    while (fgets(buffer, sizeof(buffer), pipe)) {}

    // clang-format off
    #ifdef _WIN32
    return _pclose(pipe);
    #else
    int status = pclose(pipe);
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    #endif
    // clang-format on
  }

  ScenarioResult run_scenario(nlohmann::json const& scenario, std::string const& binary_path) {
    ppforest2::io::TempFile const output;
    output.clear();

    ppforest2::io::TempFile const config_file;
    auto config = build_evaluate_config(scenario);
    ppforest2::io::json::write_file(config, config_file.path());

    std::string name = scenario["name"];
    std::string cmd  = fmt::format(
        R"("{}" -q --no-color evaluate --config "{}" -o "{}")", binary_path, config_file.path(), output.path()
    );

    auto [ret, wall_ms] = ppforest2::io::measure_time_ms([&] { return run_command(cmd); });

    user_error(ret == 0, fmt::format("Scenario '{}' failed (exit code {})", name, ret));

    return ScenarioResult(name, static_cast<double>(wall_ms), ppforest2::io::json::read_file(output.path()));
  }

  SuiteResult run_suite(BenchmarkSuite const& suite, std::string const& binary_path, io::Output& out) {
    using namespace ppforest2::io::style;

    SuiteResult result;
    result.suite_name = suite.name;
    result.timestamp  = ppforest2::io::now_iso8601();

    int total = static_cast<int>(suite.scenarios.size());

    out.indent();

    auto [_, total_ms] = ppforest2::io::measure_time_ms([&] {
      for (int i = 0; i < total; ++i) {
        auto const& scenario = suite.scenarios[i];

        out.print("[{}/{}] Running {}...", i + 1, total, emphasis(scenario["name"]));
        auto sr = run_scenario(scenario, binary_path);
        out.println("Done");

        result.results.push_back(std::move(sr));
      }

      return 0;
    });

    out.dedent();

    result.total_time_ms = static_cast<double>(total_ms);

    return result;
  }

  int run_benchmark(Params const& params, std::string const& binary_path) {
    using namespace ppforest2::io::style;

    io::Output out(params.quiet);

    auto const& bench = params.benchmark;

    auto suite = parse_suite(bench.scenarios_path);

    for (auto& s : suite.scenarios) {
      s.merge_patch(params.evaluate.overrides());
    }

    out.println("{} {} scenarios from {}", emphasis("Benchmarking"), suite.scenarios.size(), bench.scenarios_path);
    out.newline();

    // Run scenarios
    auto result = run_suite(suite, binary_path, out);

    // Load baseline for comparison
    std::optional<Baseline> baseline;

    if (!bench.baseline_path.empty()) {
      baseline.emplace(SuiteResult(io::json::read_file(bench.baseline_path, user_error)));
    }

    // Build report
    BenchmarkReport report(result, baseline);

    // Print results
    if (bench.format == "markdown") {
      report.print(out, BenchmarkReport::Markdown{});
    } else {
      report.print(out, BenchmarkReport::Text{});
    }

    // Export results
    for (auto const& path : bench.outputs) {
      if (path.size() >= 4 && path.compare(path.size() - 4, 4, ".csv") == 0) {
        io::text::write_file(report.to_csv(), path, user_error);
      } else {
        io::json::write_file(report.to_json(), path, user_error);
      }
      out.saved("Results", path);
    }

    return 0;
  }
}
