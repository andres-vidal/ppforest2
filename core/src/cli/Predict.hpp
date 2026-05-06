/**
 * @file Predict.hpp
 * @brief Predict subcommand handler.
 */
#pragma once

#include "cli/CLIOptions.hpp"

namespace CLI {
  class App;
}

namespace ppforest2::cli {
  /** @brief Register predict subcommand options on @p app. */
  void setup_predict(CLI::App& app, Params& params);

  /**
   * @brief Run the predict subcommand.
   * @return Exit code (0 on success).
   */
  int run_predict(Params const& params);
}
