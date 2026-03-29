/**
 * @file Train.cpp
 * @brief Model training utilities and train subcommand handler.
 */
#include "cli/Train.hpp"
#include "cli/Metrics.hpp"
#include "ppforest2.hpp"

#include <CLI/CLI.hpp>
#include <fmt/format.h>
#include <fstream>

#include "stats/Simulation.hpp"
#include "io/Presentation.hpp"
#include "io/Output.hpp"
#include "io/IO.hpp"
#include "io/Timing.hpp"
#include "serialization/Json.hpp"
#include "utils/UserError.hpp"

#include <nlohmann/json.hpp>

using namespace ppforest2;
using namespace ppforest2::types;
using namespace ppforest2::stats;
using json = nlohmann::json;

namespace ppforest2::cli {
  void add_model_options(CLI::App *sub, ModelParams& model) {
    sub->add_option("-t,--trees", model.trees, "Number of trees (default: 100, 0 for single tree)")
    ->check(CLI::NonNegativeNumber);
    sub->add_option("-l,--lambda", model.lambda, "Method selection (0=LDA, (0,1]=PDA)")
    ->check(CLI::Range(0.0f, 1.0f));
    sub->add_option("--threads", model.threads, "Number of threads (default: CPU cores)")
    ->check(CLI::PositiveNumber);
    sub->add_option("-r,--seed", model.seed, "Random seed (default: random)");
    sub->add_option("-v,--vars", model.vars_input, "Features per split (integer=count, decimal or fraction=proportion, default: 0.5)");
    sub->add_option("--max-retries", model.max_retries, "Max retries for degenerate trees (default: 3)")
    ->check(CLI::NonNegativeNumber);
  }

  CLI::App * setup_train(CLI::App& app, CLIOptions& params) {
    auto sub = app.add_subcommand("train", "Train a model");
    sub->add_option("-d,--data", params.data_path, "CSV training data")
    ->required()
    ->check(CLI::ExistingFile);
    add_model_options(sub, params.model);
    auto save_opt = sub->add_option("-s,--save", params.save_path, "Save trained model to JSON file (default: model.json)");
    auto no_save  = sub->add_flag("--no-save", params.no_save, "Skip saving the model (for benchmarking)");
    save_opt->excludes(no_save);
    no_save->excludes(save_opt);
    sub->add_flag("--no-metrics", params.no_metrics, "Skip variable importance computation and output");
    return sub;
  }
}

namespace ppforest2::cli {
namespace {
  json build_config_json(const CLIOptions& params) {
    json config;
    config["trees"]   = params.model.trees;
    config["lambda"]  = params.model.lambda;
    config["seed"]    = params.model.seed;
    config["threads"] = params.model.threads;

    if (params.model.trees > 0 && params.model.n_vars > 0) {
      config["vars"] = params.model.n_vars;
    }

    if (!params.data_path.empty()) {
      config["data"] = params.data_path;
    }

    return config;
  }
}

  DataPacket read_data(const CLIOptions& params, ppforest2::stats::RNG& rng) {
    if (!params.data_path.empty()) {
      try {
        return io::csv::read_sorted(params.data_path);
      } catch (const ppforest2::UserError& e) {
        fmt::print(stderr, "Error: {}\n", e.what());
        fmt::print(stderr, "File: {}\n", params.data_path);
        exit(1);
      } catch (const std::runtime_error& e) {
        fmt::print(stderr, "Error reading CSV file: {}\n", e.what());
        fmt::print(stderr, "Please ensure the file exists and is properly formatted\n");
        exit(1);
      } catch (const std::exception& e) {
        fmt::print(stderr, "Unexpected error reading file: {}\n", e.what());
        exit(1);
      }
    } else {
      SimulationParams simulation_params;
      simulation_params.mean            = params.simulation.mean;
      simulation_params.mean_separation = params.simulation.mean_separation;
      simulation_params.sd              = params.simulation.sd;

      try {
        return simulate(params.simulation.rows, params.simulation.cols, params.simulation.n_groups, rng, simulation_params);
      } catch (const std::exception& e) {
        fmt::print(stderr, "Error simulating data: {}\n", e.what());
        exit(1);
      }
    }
  }

  TrainResult train_model(
    const FeatureMatrix&   x,
    const ResponseVector&  y,
    const CLIOptions&      params,
    ppforest2::stats::RNG& rng) {
    TrainingSpec::Ptr spec;
    std::function<Model::Ptr()> fact;

    if (params.model.trees > 0) {
      spec = TrainingSpecUPDA::make(params.model.n_vars, params.model.lambda);
      fact = [&] {
        return Forest::make(*spec, x, y, params.model.trees, params.model.seed, params.model.threads, params.model.max_retries);
      };
    } else {
      spec = TrainingSpecPDA::make(params.model.lambda);
      fact = [&] {
        return Tree::make(*spec, x, y, rng);
      };
    }

    auto [model, ms] = io::measure_time_ms(fact);

    return { std::move(model), ms };
  }

  int run_train(CLIOptions& params) {
    io::Output out(params.quiet);

    // Validate save path before training
    if (!params.save_path.empty()) {
      io::check_file_not_exists(params.save_path);
    }

    ppforest2::stats::RNG rng(params.model.seed);
    auto data = read_data(params, rng);

    init_params(params, data.x.cols());

    FeatureMatrix x  = data.x;
    ResponseVector y = data.y;

    const auto train_result = train_model(x, y, params, rng);

    // Build the model JSON
    json model_json = serialization::build_model_json(
      *train_result.model, build_config_json(params),
      data.group_names, data.feature_names,
      static_cast<int>(x.rows()), static_cast<int>(x.cols()));

    model_json["training_duration_ms"] = train_result.duration;

    // Compute and add metrics to JSON
    if (!params.no_metrics) {
      compute_metrics(model_json, *train_result.model, x, y,
      data.group_names, params.model.seed);
    }

    // Save model
    if (!params.save_path.empty()) {
      io::json::write_file(model_json, params.save_path);
      model_json["save_path"] = params.save_path;
    }

    io::ConfigDisplayHints hints;
    hints.vars_percent    = params.model.trees > 0 ? static_cast<int>(params.model.p_vars * 100) : -1;
    hints.default_vars    = params.model.used_default_vars;
    hints.default_threads = params.model.used_default_threads;
    hints.default_seed    = params.model.used_default_seed;

    io::print_summary(out, model_json, hints);

    return 0;
  }
}
