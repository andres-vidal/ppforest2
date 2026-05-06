/**
 * @file Train.hpp
 * @brief Model training utilities and train subcommand handler.
 *
 * Provides shared functions used by both the train and evaluate
 * subcommands (data loading, model training, configuration display)
 * plus the run_train() entry point.
 */
#pragma once

#include "cli/CLIOptions.hpp"
#include "utils/Types.hpp"
#include "models/Model.hpp"

namespace CLI {
  class App;
}

namespace ppforest2::cli {
  /** @brief Register train subcommand options on @p app. */
  void setup_train(CLI::App& app, Params& params);

  /** @brief Add shared model options (size, lambda, threads, seed, vars) to @p sub. */
  void add_model_options(CLI::App* sub, ModelParams& model);

  /** @brief Result of a train operation containing the model and training duration. */
  struct TrainResult {
    ppforest2::Model::Ptr model;
    long long duration;
  };

  /**
   * @brief Train a single model (Forest or Tree) on the given dataset.
   *
   * Takes `x` / `y` by const ref and copies internally — regression training
   * (ByCutpoint) permutes rows in place via `std::stable_sort`, which would
   * leak across calls if we passed the caller's matrix straight through.
   * Centralising the copy here keeps the mutation contract internal and lets
   * callers (warmup loops, multi-iteration evaluate, etc.) reuse the same
   * input across calls without per-call boilerplate.
   */
  TrainResult
  train_model(ppforest2::types::FeatureMatrix const& x, ppforest2::types::OutcomeVector const& y, Params const& params);

  /**
   * @brief Run the train subcommand.
   * @return Exit code (0 on success).
   */
  int run_train(Params& params);
}
