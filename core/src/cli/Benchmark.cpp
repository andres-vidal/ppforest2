/**
 * @file Benchmark.cpp
 * @brief Benchmark scenario parsing, validation, and subprocess execution.
 */
#include "cli/Benchmark.hpp"
#include "cli/BenchmarkReport.hpp"
#include "cli/CLIOptions.hpp"
#include "cli/VarsSpec.hpp"

#include <CLI/CLI.hpp>
#include "io/TempFile.hpp"
#include "io/Timing.hpp"
#include "io/Color.hpp"
#include "io/Output.hpp"
#include "utils/Invariant.hpp"

#include <fmt/format.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <optional>
#include <set>

#ifdef _WIN32
#include <process.h>
#else
#include <sys/wait.h>
#endif

namespace ppforest2::cli {
  CLI::App * setup_benchmark(CLI::App& app, CLIOptions& params) {
    auto sub = app.add_subcommand("benchmark", "Run performance benchmarks across scenarios");
    sub->add_option("-s,--scenarios", params.benchmark.scenarios_path, "JSON scenarios file")
    ->check(CLI::ExistingFile);
    sub->add_option("-b,--baseline", params.benchmark.baseline_path, "Baseline results JSON for comparison")
    ->check(CLI::ExistingFile);
    sub->add_option("-o,--output", params.benchmark.output, "Save results to JSON file");
    sub->add_option("--csv", params.benchmark.csv, "Save results to CSV file");
    sub->add_option("-i,--iterations", params.benchmark.iterations, "Override iteration count (forces fixed mode)")
    ->check(CLI::PositiveNumber);
    sub->add_option("-p,--train-ratio", params.benchmark.train_ratio, "Override train set ratio for all scenarios")
    ->check(CLI::Range(0.01f, 0.99f));
    sub->add_option("--format", params.benchmark.format, "Output format (table, markdown)")
    ->check(CLI::IsMember({ "table", "markdown" }));
    return sub;
  }

namespace {
  /**
   * @brief Read CSV dimensions (rows, columns, unique values in last column).
   *
   * Lightweight: reads the file line by line without loading it into memory.
   * Assumes the last column is the response variable.
   */
  struct CsvDimensions { int n = 0; int p = 0; int g = 0; };

  CsvDimensions read_csv_dimensions(const std::string& path) {
    std::ifstream file(path);

    invariant(file.is_open(), fmt::format("Cannot open data file: {}", path));

    CsvDimensions dims;
    std::string line;
    int total_cols = 0;
    std::set<std::string> group_labels;

    while (std::getline(file, line)) {
      if (line.empty()) continue;

      int cols = 1;

      for (char c : line) {
        if (c == ',') ++cols;
      }

      if (total_cols == 0) {
        total_cols = cols;
      }

      // Extract last column value (response)
      auto last_comma   = line.rfind(',');
      std::string label = (last_comma != std::string::npos)
        ? line.substr(last_comma + 1)
        : line;

      group_labels.insert(label);
      ++dims.n;
    }

    dims.p = total_cols - 1;  // Last column is response
    dims.g = static_cast<int>(group_labels.size());

    return dims;
  }

  /**
   * @brief Parse a ConvergenceCriteria from a JSON object.
   *
   * Missing fields retain their struct defaults. See ConvergenceCriteria
   * in Benchmark.hpp for the full algorithm description.
   */
  ConvergenceCriteria parse_convergence(const nlohmann::json& j) {
    ConvergenceCriteria c;

    if (j.contains("cv")) c.cv = j["cv"].get<float>();

    if (j.contains("window")) c.window = j["window"].get<int>();

    if (j.contains("min")) c.min = j["min"].get<int>();

    if (j.contains("max")) c.max = j["max"].get<int>();

    return c;
  }

  /**
   * @brief Merge a scenario JSON object onto a base Scenario.
   *
   * Starts from @p defaults and overwrites only the fields present in
   * @p scenario_json. Integer vars counts are converted to proportions
   * using the scenario's @c p value (must already be set).
   */
  Scenario apply_defaults(const nlohmann::json& scenario_json, const Scenario& defaults) {
    Scenario s = defaults;

    if (scenario_json.contains("name")) s.name = scenario_json["name"].get<std::string>();

    if (scenario_json.contains("data")) s.data = scenario_json["data"].get<std::string>();

    if (scenario_json.contains("n")) s.n = scenario_json["n"].get<int>();

    if (scenario_json.contains("p")) s.p = scenario_json["p"].get<int>();

    if (scenario_json.contains("g")) s.g = scenario_json["g"].get<int>();

    if (scenario_json.contains("trees")) s.trees = scenario_json["trees"].get<int>();

    if (scenario_json.contains("vars")) {
      auto spec = ppforest2::cli::parse_vars(scenario_json["vars"]);

      if (spec.is_proportion) {
        s.vars = spec.value;
      } else {
        invariant(s.p > 0, "p must be set before using integer vars count");
        s.vars = spec.value / s.p;
      }
    }

    if (scenario_json.contains("lambda")) s.lambda = scenario_json["lambda"].get<float>();

    if (scenario_json.contains("threads")) s.threads = scenario_json["threads"].get<int>();

    if (scenario_json.contains("train_ratio")) s.train_ratio = scenario_json["train_ratio"].get<float>();

    if (scenario_json.contains("seed")) s.seed = scenario_json["seed"].get<int>();

    if (scenario_json.contains("warmup")) s.warmup = scenario_json["warmup"].get<int>();

    if (scenario_json.contains("iterations")) s.iterations = scenario_json["iterations"].get<int>();

    if (scenario_json.contains("convergence")) {
      s.convergence = parse_convergence(scenario_json["convergence"]);
    }

    return s;
  }

  /**
   * @brief Validate that all scenario fields are within legal ranges.
   * @throws std::runtime_error (via invariant) with a descriptive message
   *         naming the offending scenario and field.
   */
  void validate_scenario(const Scenario& s) {
    invariant(!s.name.empty(), "Scenario name is required");

    if (s.data.empty()) {
      // Simulation mode: n, p, g are required
      invariant(s.n > 0, fmt::format("Scenario '{}': n must be positive", s.name));
      invariant(s.p > 0, fmt::format("Scenario '{}': p must be positive", s.name));
      invariant(s.g > 1, fmt::format("Scenario '{}': g must be > 1", s.name));
    }

    invariant(s.trees >= 0, fmt::format("Scenario '{}': trees must be >= 0", s.name));
    invariant(s.vars > 0 && s.vars <= 1, fmt::format("Scenario '{}': vars must be in (0, 1]", s.name));
    invariant(s.lambda >= 0 && s.lambda <= 1, fmt::format("Scenario '{}': lambda must be in [0, 1]", s.name));
    invariant(s.train_ratio > 0 && s.train_ratio < 1, fmt::format("Scenario '{}': train_ratio must be in (0, 1)", s.name));
    invariant(s.warmup >= 0, fmt::format("Scenario '{}': warmup must be >= 0", s.name));
  }

  /**
   * @brief Build the `ppforest2 evaluate` shell command for a scenario.
   *
   * Constructs the full command string including simulated data shape,
   * model parameters, and iteration/convergence flags. Always runs
   * with @c -q and @c --no-color for clean JSON output.
   *
   * @param s            The scenario to evaluate.
   * @param binary_path  Path to the ppforest2 binary.
   * @param output_path  Where the subprocess writes its JSON results.
   */
  std::string build_evaluate_command(
    const Scenario&    s,
    const std::string& binary_path,
    const std::string& output_path) {
    std::string cmd;

    if (s.data.empty()) {
      cmd = fmt::format(
        "\"{}\" -q --no-color evaluate --simulate {}x{}x{} -r {} -p {} -o \"{}\"",
        binary_path, s.n, s.p, s.g, s.seed, s.train_ratio, output_path);
    } else {
      cmd = fmt::format(
        "\"{}\" -q --no-color evaluate --data \"{}\" -r {} -p {} -o \"{}\"",
        binary_path, s.data, s.seed, s.train_ratio, output_path);
    }

    if (s.trees > 0) {
      cmd += fmt::format(" -t {} -v {}", s.trees, s.vars);
    } else {
      cmd += " -t 0";
    }

    cmd += fmt::format(" -l {}", s.lambda);

    if (s.threads > 0) {
      cmd += fmt::format(" --threads {}", s.threads);
    }

    if (s.warmup > 0) {
      cmd += fmt::format(" --warmup {}", s.warmup);
    }

    if (s.iterations > 0) {
      // Fixed iteration mode (-i disables convergence)
      cmd += fmt::format(" -i {}", s.iterations);
    } else {
      // Convergence mode (default)
      cmd += fmt::format(" --convergence-cv {} --convergence-max {}", s.convergence.cv, s.convergence.max);
    }

    return cmd;
  }

  /**
   * @brief Read and parse the JSON output written by `ppforest2 evaluate`.
   *
   * Copies the scenario's data-shape fields (n, p, g, trees, vars) into
   * the result for reporting, then extracts timing and error metrics.
   *
   * @param path  Path to the JSON file written by the subprocess.
   * @param s     The originating scenario (for metadata).
   * @throws std::runtime_error if the file cannot be opened.
   */
  ScenarioResult parse_evaluate_output(const std::string& path, const Scenario& s) {
    std::ifstream file(path);

    invariant(file.is_open(), fmt::format("Failed to read results for scenario '{}'", s.name));

    auto j = nlohmann::json::parse(file);

    ScenarioResult result;
    result.name = s.name;
    result.data = s.data;

    if (s.data.empty()) {
      result.n = s.n;
      result.p = s.p;
      result.g = s.g;
    } else {
      auto dims = read_csv_dimensions(s.data);
      result.n = dims.n;
      result.p = dims.p;
      result.g = dims.g;
    }

    result.trees       = s.trees;
    result.vars        = s.vars;
    result.train_ratio = s.train_ratio;

    result.runs           = j.value("runs", 0);
    result.mean_time_ms   = j.value("mean_time_ms", 0.0);
    result.std_time_ms    = j.value("std_time_ms", 0.0);
    result.mean_tr_error  = j.value("mean_train_error", 0.0);
    result.mean_te_error  = j.value("mean_test_error", 0.0);
    result.peak_rss_bytes = j.value("peak_rss_bytes", (long)-1);
    result.peak_rss_mb    = j.value("peak_rss_mb", -1.0);

    return result;
  }
}

  BenchmarkSuite parse_suite(const std::string& path) {
    std::ifstream file(path);

    invariant(file.is_open(), fmt::format("Failed to open scenarios file: {}", path));

    nlohmann::json j;

    try {
      j = nlohmann::json::parse(file);
    } catch (const nlohmann::json::parse_error& e) {
      invariant(false, fmt::format("Invalid JSON in scenarios file: {}", e.what()));
    }

    return parse_suite(j);
  }

  BenchmarkSuite parse_suite(const nlohmann::json& j) {
    BenchmarkSuite suite;

    if (j.contains("name")) {
      suite.name = j["name"].get<std::string>();
    }

    // Parse defaults
    Scenario defaults;

    if (j.contains("defaults")) {
      defaults = apply_defaults(j["defaults"], defaults);
    }

    // Parse scenarios
    invariant(j.contains("scenarios") && j["scenarios"].is_array(), "Scenarios file must contain a 'scenarios' array");

    for (const auto& scenario_json : j["scenarios"]) {
      Scenario s = apply_defaults(scenario_json, defaults);
      validate_scenario(s);
      suite.scenarios.push_back(std::move(s));
    }

    invariant(!suite.scenarios.empty(), "Scenarios array must not be empty");

    return suite;
  }

  SuiteResult parse_results(const std::string& path) {
    std::ifstream file(path);

    invariant(file.is_open(), fmt::format("Failed to open baseline results file: {}", path));

    auto j = nlohmann::json::parse(file);

    SuiteResult result;
    result.suite_name    = j.value("suite_name", "");
    result.timestamp     = j.value("timestamp", "");
    result.total_time_ms = j.value("total_time_ms", 0.0);

    if (j.contains("results") && j["results"].is_array()) {
      for (const auto& r : j["results"]) {
        ScenarioResult sr;
        sr.name           = r.value("name", "");
        sr.data           = r.value("data", "");
        sr.n              = r.value("n", 0);
        sr.p              = r.value("p", 0);
        sr.g              = r.value("g", 0);
        sr.trees          = r.value("trees", 0);
        sr.vars           = r.value("vars", 0.0f);
        sr.train_ratio    = r.value("train_ratio", 0.7f);
        sr.runs           = r.value("runs", 0);
        sr.mean_time_ms   = r.value("mean_time_ms", 0.0);
        sr.std_time_ms    = r.value("std_time_ms", 0.0);
        sr.mean_tr_error  = r.value("mean_train_error", 0.0);
        sr.mean_te_error  = r.value("mean_test_error", 0.0);
        sr.peak_rss_bytes = r.value("peak_rss_bytes", (long)-1);
        sr.peak_rss_mb    = r.value("peak_rss_mb", -1.0);

        result.results.push_back(std::move(sr));
      }
    }

    return result;
  }

  nlohmann::json SuiteResult::to_json() const {
    nlohmann::json j;
    j["suite_name"]    = suite_name;
    j["timestamp"]     = timestamp;
    j["total_time_ms"] = total_time_ms;

    nlohmann::json results_array = nlohmann::json::array();

    for (const auto& r : results) {
      nlohmann::json rj;
      rj["name"] = r.name;

      if (!r.data.empty()) {
        rj["data"] = r.data;
      }

      rj["n"]                = r.n;
      rj["p"]                = r.p;
      rj["g"]                = r.g;
      rj["trees"]            = r.trees;
      rj["vars"]             = r.vars;
      rj["train_ratio"]      = r.train_ratio;
      rj["runs"]             = r.runs;
      rj["mean_time_ms"]     = r.mean_time_ms;
      rj["std_time_ms"]      = r.std_time_ms;
      rj["mean_train_error"] = r.mean_tr_error;
      rj["mean_test_error"]  = r.mean_te_error;

      if (r.peak_rss_bytes >= 0) {
        rj["peak_rss_bytes"] = r.peak_rss_bytes;
        rj["peak_rss_mb"]    = r.peak_rss_mb;
      }

      rj["scenario_time_ms"] = r.scenario_time_ms;

      results_array.push_back(rj);
    }

    j["results"] = results_array;
    return j;
  }

  /**
   * @brief Run a command and return its exit code.
   *
   * Uses popen/pclose instead of std::system() to get reliable
   * exit codes on all platforms.  On Windows, std::system() wraps
   * via cmd.exe which can return unexpected exit codes.
   */
  int run_command(const std::string& cmd) {
    #ifdef _WIN32
    std::string full = "\"" + cmd + "\" 2>NUL";
    FILE *pipe       = _popen(full.c_str(), "r");
    #else
    std::string full = cmd + " 2>/dev/null";
    FILE *pipe       = popen(full.c_str(), "r");
    #endif

    if (!pipe) return -1;

    // Drain the pipe (stdout is empty due to -q, but must be consumed)
    char buffer[4096];

    while (fgets(buffer, sizeof(buffer), pipe)) {
    }

    #ifdef _WIN32
    return _pclose(pipe);

    #else
    int status = pclose(pipe);
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;

    #endif
  }

  ScenarioResult run_scenario(
    const Scenario&    scenario,
    const std::string& binary_path,
    bool               quiet) {
    ppforest2::io::TempFile output;
    output.clear();
    std::string cmd = build_evaluate_command(scenario, binary_path, output.path());

    auto [ret, wall_ms] = ppforest2::io::measure_time_ms([&] {
      return run_command(cmd);
    });

    invariant(ret == 0, fmt::format("Scenario '{}' failed (exit code {})", scenario.name, ret));

    ScenarioResult result = parse_evaluate_output(output.path(), scenario);
    result.scenario_time_ms = wall_ms;
    return result;
  }

  SuiteResult run_suite(
    const BenchmarkSuite& suite,
    const std::string&    binary_path,
    bool                  quiet,
    ProgressCallback      progress) {
    SuiteResult result;
    result.suite_name = suite.name;
    result.timestamp  = ppforest2::io::now_iso8601();

    int total = static_cast<int>(suite.scenarios.size());

    auto [_, total_ms] = ppforest2::io::measure_time_ms([&] {
      for (int i = 0; i < total; ++i) {
        const auto& scenario = suite.scenarios[i];

        if (progress) {
          progress(i, total, scenario.name);
        }

        auto sr = run_scenario(scenario, binary_path, quiet);
        result.results.push_back(std::move(sr));
      }

      if (progress) {
        progress(total, total, "");
      }

      return 0;
    });

    result.total_time_ms = total_ms;

    return result;
  }

  int run_benchmark(CLIOptions& params, const std::string& binary_path) {
    using namespace ppforest2::io::style;

    io::Output out(params.quiet);

    // Load scenarios
    if (params.benchmark.scenarios_path.empty()) {
      out.errorln("{} No scenarios file specified. Use -s/--scenarios <file>", error("Error:"));
      return 1;
    }

    BenchmarkSuite suite;

    if (int rc = out.try_or_fail([&] {
      suite = parse_suite(params.benchmark.scenarios_path);
    })) {
      return rc;
    }

    // Apply overrides
    if (params.benchmark.iterations > 0) {
      for (auto& s : suite.scenarios) {
        s.iterations = params.benchmark.iterations;
      }
    }

    if (params.benchmark.train_ratio > 0) {
      for (auto& s : suite.scenarios) {
        s.train_ratio = params.benchmark.train_ratio;
      }
    }

    out.println("{} {} scenarios from {}", emphasis("Benchmarking"), suite.scenarios.size(), params.benchmark.scenarios_path);
    out.newline();

    // Run scenarios
    SuiteResult result;

    if (int rc = out.try_or_fail([&] {
      result = run_suite(suite, binary_path, params.quiet,
      [&](int idx, int total, const std::string& name) {
        if (idx < total) {
          out.indent();
          out.println("[{}/{}] Running {}...", idx + 1, total, emphasis(name));
          out.dedent();
        }
      });
    })) {
      return rc;
    }

    // Load baseline for comparison
    std::optional<SuiteResult> baseline;

    if (!params.benchmark.baseline_path.empty()) {
      if (int rc = out.try_or_fail([&] {
        baseline = parse_results(params.benchmark.baseline_path);
      },
      "Failed to load baseline")) return rc;
    }

    // Print results
    if (params.benchmark.format == "markdown") {
      io::Output md_out(false);
      print_benchmark_markdown(md_out, result, baseline);
    } else {
      print_benchmark_table(out, result, baseline);
    }

    // Export results
    if (!params.benchmark.output.empty()) {
      if (int rc = out.try_or_fail([&] {
        write_results_json(result, params.benchmark.output);
      })) return rc;

      out.saved("Results", params.benchmark.output);
    }

    if (!params.benchmark.csv.empty()) {
      if (int rc = out.try_or_fail([&] {
        write_results_csv(result, params.benchmark.csv);
      })) return rc;

      out.saved("CSV", params.benchmark.csv);
    }

    return 0;
  }
}
