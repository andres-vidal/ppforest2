/**
 * @file CLIOptions.cpp
 * @brief CLI argument parsing, validation, and configuration for ppforest2.
 */
#include "cli/CLIOptions.hpp"
#include "cli/Train.hpp"
#include "cli/Predict.hpp"
#include "cli/Evaluate.hpp"
#include "cli/Benchmark.hpp"
#include "cli/Summarize.hpp"
#include "cli/VarsSpec.hpp"

#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>
#include <fmt/format.h>
#include <random>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <functional>
#include <set>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ppforest2::cli {
namespace {
  /**
   * @brief JSON config file adapter for CLI11.
   *
   * Reads and writes CLI11 options from/to JSON files.
   * Supports broadcasting top-level keys to all subcommands
   * listed in `subcommand_names_`.
   */
  class ConfigJSON : public CLI::Config {
    public:
      ConfigJSON(std::vector<std::string> subcommand_names = {})
        : subcommand_names_(std::move(subcommand_names)) {
      }

      std::string to_config(const CLI::App * app, bool default_also, bool, std::string) const override {
        nlohmann::json j;

        for (const CLI::Option *opt : app->get_options({})) {
          if (!opt->get_lnames().empty() && opt->get_configurable()) {
            std::string name = opt->get_lnames()[0];

            if (opt->get_type_size() != 0) {
              if (opt->count() == 1) j[name] = opt->results().at(0);
              else if (opt->count() > 1) j[name] = opt->results();
              else if (default_also && !opt->get_default_str().empty()) j[name] = opt->get_default_str();
            } else if (opt->count() == 1) {
              j[name] = true;
            } else if (opt->count() > 1) {
              j[name] = opt->count();
            } else if (opt->count() == 0 && default_also) {
              j[name] = false;
            }
          }
        }

        for (const CLI::App *subcom : app->get_subcommands({})) {
          j[subcom->get_name()] = nlohmann::json(to_config(subcom, default_also, false, ""));
        }

        return j.dump(4);
      }

      std::vector<CLI::ConfigItem> from_config(std::istream &input) const override {
        nlohmann::json j;
        input >> j;
        return _from_config(j);
      }

    private:
      std::vector<std::string> subcommand_names_;

      std::vector<CLI::ConfigItem>
      _from_config(nlohmann::json j, std::string name = "", std::vector<std::string> prefix = {}) const {
        std::vector<CLI::ConfigItem> results;

        if (j.is_object()) {
          for (auto item = j.begin(); item != j.end(); ++item) {
            auto copy_prefix = prefix;

            if (!name.empty()) copy_prefix.push_back(name);

            // Normalize: accept both snake_case and kebab-case keys
            std::string key = item.key();
            std::replace(key.begin(), key.end(), '_', '-');

            auto sub_results = _from_config(*item, key, copy_prefix);
            results.insert(results.end(), sub_results.begin(), sub_results.end());
          }
        } else if (!name.empty()) {
          if (prefix.empty() && !subcommand_names_.empty()) {
            for (const auto& sub_name : subcommand_names_) {
              results.emplace_back();
              CLI::ConfigItem &res = results.back();
              res.name    = name;
              res.parents = { sub_name };
              _set_inputs(res, j);
            }
          } else {
            results.emplace_back();
            CLI::ConfigItem &res = results.back();
            res.name    = name;
            res.parents = prefix;
            _set_inputs(res, j);
          }
        }

        return results;
      }

      void _set_inputs(CLI::ConfigItem &res, const nlohmann::json &j) const {
        if (j.is_boolean()) {
          res.inputs = { j.get<bool>() ? "true" : "false" };
        } else if (j.is_number()) {
          std::stringstream ss;
          ss << j.get<double>();
          res.inputs = { ss.str() };
        } else if (j.is_string()) {
          res.inputs = { j.get<std::string>() };
        } else if (j.is_array()) {
          for (const auto &val : j) {
            res.inputs.push_back(val.get<std::string>());
          }
        } else {
          throw CLI::ConversionError("Failed to convert " + res.name);
        }
      }
  };

  /**
   * @brief Warn about unrecognized keys in a JSON config file.
   *
   * CLI11's built-in config_extras_mode doesn't work with the
   * broadcast pattern used by ConfigJSON (top-level keys forwarded
   * to all subcommands), so we validate keys manually by walking
   * the config JSON and checking against registered option names.
   */
  void warn_unknown_config_keys(io::Output& out, const CLI::App& app) {
    auto *config_opt = app.get_config_ptr();

    if (!config_opt || config_opt->count() == 0) return;

    std::string config_path = config_opt->results().at(0);
    std::ifstream file(config_path);

    if (!file.is_open()) return;

    nlohmann::json j;

    try {
      file >> j;
    } catch (...) {
      return;
    }

    if (!j.is_object()) return;

    // Collect all known option names and subcommand names
    std::set<std::string> known;

    for (const CLI::Option *opt : app.get_options({})) {
      for (const auto& name : opt->get_lnames()) {
        known.insert(name);
      }
    }

    for (const CLI::App *sub : app.get_subcommands({})) {
      known.insert(sub->get_name());

      for (const CLI::Option *opt : sub->get_options({})) {
        for (const auto& name : opt->get_lnames()) {
          known.insert(name);
        }
      }
    }

    // Walk config JSON and warn about unknown keys
    std::function<void(const nlohmann::json&, const std::string&)> check;
    check = [&](const nlohmann::json& obj, const std::string& context) {
      for (auto it = obj.begin(); it != obj.end(); ++it) {
        std::string key = it.key();
        std::replace(key.begin(), key.end(), '_', '-');

        if (known.find(key) == known.end()) {
          std::string full_key = context.empty() ? it.key() : context + "." + it.key();
          out.println("Warning: Unknown config key '{}' — ignoring", full_key);
        } else if (it->is_object()) {
          check(*it, key);
        }
      }
    };

    check(j, "");
  }

  void post_parse(CLIOptions & params, CLI::App& app) {
    auto *train_sub     = app.get_subcommand("train");
    auto *predict_sub   = app.get_subcommand("predict");
    auto *eval_sub      = app.get_subcommand("evaluate");
    auto *bench_sub     = app.get_subcommand("benchmark");
    auto *summarize_sub = app.get_subcommand("summarize");

    // Determine subcommand
    if (train_sub->parsed()) {
      params.subcommand = Subcommand::train;
    } else if (predict_sub->parsed()) {
      params.subcommand = Subcommand::predict;
    } else if (eval_sub->parsed()) {
      params.subcommand = Subcommand::evaluate;
    } else if (bench_sub->parsed()) {
      params.subcommand = Subcommand::benchmark;
    } else if (summarize_sub->parsed()) {
      params.subcommand = Subcommand::summarize;
    }

    // Handle --no-save for train
    if (params.subcommand == Subcommand::train && params.no_save) {
      params.save_path.clear();
    }

    // Evaluate has no --save
    if (params.subcommand == Subcommand::evaluate) {
      params.save_path.clear();
    }

    // Interpret --vars input
    if (!params.model.vars_input.empty()) {
      try {
        auto spec = parse_vars(params.model.vars_input);

        if (spec.is_proportion) {
          params.model.p_vars = spec.value;
        } else {
          params.model.n_vars = static_cast<int>(spec.value);
        }
      } catch (const std::exception& e) {
        fmt::print(stderr, "Error: Invalid --vars value: {}\n", e.what());
        std::exit(1);
      }
    }

    // Validate simulate format
    if (!params.simulation.format.empty()) {
      std::string sim_str = params.simulation.format;
      size_t x1           = sim_str.find('x');
      size_t x2           = sim_str.find('x', x1 + 1);

      if (x1 == std::string::npos || x2 == std::string::npos) {
        fmt::print(stderr, "Error: Simulate format must be NxMxK (e.g., 1000x10x2)\n");
        std::exit(1);
      }

      try {
        params.simulation.rows     = std::stoi(sim_str.substr(0, x1));
        params.simulation.cols     = std::stoi(sim_str.substr(x1 + 1, x2 - x1 - 1));
        params.simulation.n_groups = std::stoi(sim_str.substr(x2 + 1));

        if (params.simulation.rows <= 0 || params.simulation.cols <= 0 || params.simulation.n_groups <= 1) {
          throw std::out_of_range("Values must be positive and groups must be > 1");
        }
      } catch (const std::exception& e) {
        fmt::print(stderr, "Error: Invalid simulate values: {}\n", e.what());
        std::exit(1);
      }
    }

    // Evaluate requires data source
    if (params.subcommand == Subcommand::evaluate && params.simulation.format.empty() && params.data_path.empty()) {
      fmt::print(stderr, "Error: Must specify either --simulate or --data\n");
      std::exit(1);
    }

    // -i disables convergence; without -i, convergence is on
    if (params.subcommand == Subcommand::evaluate) {
      if (eval_sub->get_option("--iterations")->count() > 0) {
        params.convergence.enabled = false;
      }
    }

    io::Output out(params.quiet);
    warn_unknown_config_keys(out, app);
    warn_unused_params(out, params);
  }
}

  void warn_unused_params(io::Output& out, const CLIOptions& params) {
    if (params.model.trees == 0) {
      bool has_warnings = false;

      if (params.model.threads != -1) {
        out.println("Warning: threads parameter is ignored when training a single tree");
        has_warnings = true;
      }

      if (params.model.p_vars != -1 || params.model.n_vars != -1) {
        out.println("Warning: --vars parameter is ignored when training a single tree");
        has_warnings = true;
      }

      if (has_warnings) {
        out.println("Single trees always use all features for splitting");
      }
    }
  }

  void init_params(CLIOptions& params, int total_vars) {
    if (params.model.lambda == -1) {
      params.model.lambda = 0.5;
    }

    if (params.evaluate.train_ratio <= 0 || params.evaluate.train_ratio >= 1) {
      fmt::print(stderr, "Error: Train ratio must be between 0 and 1\n");
      exit(1);
    }

    if (params.model.seed == -1) {
      std::random_device rd;
      params.model.seed              = rd();
      params.model.used_default_seed = true;
    }

    if (params.model.threads == -1) {
      #ifdef _OPENMP
      params.model.threads = omp_get_max_threads();
      #else
      params.model.threads = 1;
      #endif
      params.model.used_default_threads = true;
    }

    if (total_vars > 0 && params.model.trees > 0) {
      if (params.model.n_vars != -1) {
        params.model.p_vars = static_cast<float>(params.model.n_vars) / total_vars;
      } else if (params.model.p_vars != -1) {
        params.model.n_vars = std::round(total_vars * params.model.p_vars);
      } else {
        params.model.p_vars            = 0.5;
        params.model.n_vars            = std::round(total_vars * params.model.p_vars);
        params.model.used_default_vars = true;
      }
    }
  }

  CLIOptions parse_args(int argc, char *argv[]) {
    CLIOptions params;

    CLI::App app{ "ppforest2 - Projection Pursuit Trees and Forests" };
    app.require_subcommand(1);
    app.fallthrough();
    app.set_version_flag("--version,-V", PPFOREST2_VERSION, "Print version and exit");

    // Global options
    app.add_flag("--quiet,-q", params.quiet, "Suppress all terminal output");
    app.add_flag("--no-color", params.no_color, "Disable colored output");
    app.config_formatter(std::make_shared<ConfigJSON>(std::vector<std::string>{ "train", "predict", "evaluate", "benchmark", "summarize" }));
    app.set_config("--config", "", "Read parameters from JSON config file");

    // Subcommands
    setup_train(app, params);
    setup_predict(app, params);
    setup_evaluate(app, params);
    setup_benchmark(app, params);
    setup_summarize(app, params);

    // Parse
    try {
      app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
      std::exit(app.exit(e));
    }

    post_parse(params, app);
    return params;
  }
}
