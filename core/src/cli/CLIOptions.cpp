/**
 * @file CLIOptions.cpp
 * @brief CLI argument parsing, validation, and configuration for ppforest2.
 */
#include "cli/CLIOptions.hpp"
#include "cli/Benchmark.hpp"
#include "cli/Evaluate.hpp"
#include "cli/Predict.hpp"
#include "cli/Serve.hpp"
#include "cli/Summarize.hpp"
#include "cli/Train.hpp"
#include "io/IO.hpp"
#include "utils/UserError.hpp"

#include <CLI/CLI.hpp>
#include <fmt/format.h>
#include <fstream>
#include <nlohmann/json.hpp>
#include <random>
#include <string>

namespace ppforest2::cli {

  void Params::resolve() {
    model.resolve();
    simulation.resolve_format();
  }

  Params::Params(nlohmann::json const& config)
      : model(config)
      , simulation(config)
      , evaluate(config) {
    data_path = config.value("data", data_path);
  }

  namespace {
    /**
     * @brief Pre-scan argv for --config <path>.
     *
     * Extracts the config path before CLI11 parses so the config
     * can be loaded first and CLI args override it naturally.
     */
    std::string find_config_path(int argc, char* argv[]) {
      for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == "--config") {
          return argv[i + 1];
        }
      }
      return {};
    }

    /** @brief Read a JSON config file into a Params, or return defaults if path is
     * empty. */
    Params load_config(std::string const& path) {
      if (path.empty()) {
        return {};
      }

      io::check_file_exists(path);

      std::ifstream file(path);
      user_error(file.is_open(), fmt::format("Cannot open config file: {}", path));

      nlohmann::json config;
      try {
        file >> config;
      } catch (std::exception const& e) {
        throw ppforest2::UserError(fmt::format("Invalid JSON in config file: {}", e.what()));
      }

      if (!config.is_object()) {
        return {};
      }

      try {
        return Params(config);
      } catch (nlohmann::json::exception const& e) {
        // `Params(config)` reads typed values (`config.value(...)`), which
        // throws on a wrong-typed field — surface it as the user error it is.
        throw ppforest2::UserError(fmt::format("Invalid config file '{}': {}", path, e.what()));
      }
    }

  }

  void warn_unused_params(io::Output& out, Params const& params) {
    if (params.model.size == 0) {
      bool has_warnings = false;

      if (params.model.threads.has_value()) {
        out.println("Warning: threads parameter is ignored when training a single tree");
        has_warnings = true;
      }

      if (params.model.p_vars.has_value() || params.model.n_vars.has_value()) {
        out.println("Warning: --n-vars/--p-vars parameter is ignored when training a single tree");
        has_warnings = true;
      }

      if (has_warnings) {
        out.println("Single trees always use all features for splitting");
      }
    }
  }

  void Params::resolve_seed() {
    if (!model.seed) {
      std::random_device rd;
      model.seed = static_cast<int>(rd());
    }
  }

  void Params::resolve_defaults(unsigned int total_vars) {
    model.resolve_defaults(total_vars);
  }

  stats::DataPacket Params::read_data(stats::RNG& rng) {
    if (!data_path.empty()) {
      if (model.mode_input.empty()) {
        // No --mode: detect it from the y column's written form; empty
        // group_names means regression shape.
        stats::DataPacket data = io::csv::read_sorted(data_path);
        model.mode_input       = data.group_names.empty() ? "regression" : "classification";
        return data;
      }

      // --mode given (CLI or config file): parse the response in that mode
      // instead of trusting the written form — integer-written regression
      // responses must stay numeric, not become label codes.
      return io::csv::read_sorted(data_path, types::mode_from_string(model.mode_input));
    }

    // Simulated data: no CSV to auto-detect from, so default to classification
    // when --mode wasn't passed.
    if (model.mode_input.empty()) {
      model.mode_input = "classification";
    }

    return simulation.simulate(types::mode_from_string(model.mode_input), rng);
  }

  nlohmann::json Params::to_json() const {
    nlohmann::json config;

    if (!data_path.empty()) {
      config["data"] = data_path;
    }
    if (!simulation.format.empty()) {
      config["simulate"] = simulation.format;
    }

    config.update(model.to_json());
    config.update(evaluate.to_json());

    return config;
  }

  Params parse_args(int argc, char* argv[]) {
    // Load config first so it serves as defaults; CLI args override naturally.
    Params params = load_config(find_config_path(argc, argv));

    CLI::App app{"ppforest2 - Projection Pursuit Trees and Forests"};
    app.require_subcommand(1);
    app.fallthrough();
    app.set_version_flag("--version,-V", PPFOREST2_VERSION, "Print version and exit");

    // Global options
    std::string config_sink; // consumed by find_config_path; registered so --help
                             // shows it
    app.add_flag("--quiet,-q", params.quiet, "Suppress all terminal output");
    app.add_flag("--no-color", params.no_color, "Disable colored output");
    app.add_option("--config", config_sink, "Read parameters from JSON config file");

    // Subcommands (each sets params.subcommand via callback)
    setup_train(app, params);
    setup_predict(app, params);
    setup_evaluate(app, params);
    setup_benchmark(app, params);
    setup_summarize(app, params);
    setup_serve(app, params);

    // Parse — CLI11 throws for both errors and --help/--version.
    // Success (help/version) prints and exits; errors become UserError.
    try {
      app.parse(argc, argv);
    } catch (CLI::Success const& e) {
      std::exit(app.exit(e));
    } catch (CLI::ParseError const& e) {
      throw ppforest2::UserError(e.what());
    }

    // Check file existence for data path (can come from config).
    // CLI-exclusive paths (model, scenarios, baseline) are validated
    // by CLI11 ExistingFile checks in their setup functions.
    if (!params.data_path.empty()) {
      io::check_file_exists(params.data_path);
    }

    // Resolve intermediate representations (strategy strings, p_vars, simulate format)
    params.resolve();

    return params;
  }
}
