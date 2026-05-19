/**
 * @file Serve.hpp
 * @brief Serve subcommand handler: HTTP server around a loaded model.
 */
#pragma once

#include "cli/CLIOptions.hpp"

namespace CLI {
  class App;
}

namespace ppforest2::cli {
  /** @brief Register serve subcommand options on @p app. */
  void setup_serve(CLI::App& app, Params& params);

  /**
   * @brief Run the serve subcommand.
   * @return Exit code (0 on graceful shutdown, non-zero on bind failure).
   */
  int run_serve(Params const& params);
}
