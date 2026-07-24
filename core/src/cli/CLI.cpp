/**
 * @file CLI.cpp
 * @brief Main entry point for the ppforest2 command-line tool.
 */
#include "cli/Benchmark.hpp"
#include "cli/CLIOptions.hpp"
#include "cli/Evaluate.hpp"
#include "cli/Predict.hpp"
#include "cli/Serve.hpp"
#include "cli/Summarize.hpp"
#include "cli/Train.hpp"
#include "io/Color.hpp"
#include "io/IO.hpp"
#include "utils/UserError.hpp"

#include <fmt/format.h>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace ppforest2::cli;

int main(int argc, char* argv[]) {
  try {
    Params params = parse_args(argc, argv);

    ppforest2::io::style::init_color(params.no_color);

    // Post-parse: ensure .json extension on output paths
    if (!params.save_path.empty()) {
      params.save_path = ppforest2::io::json::ensure_extension(params.save_path);
    }

    if (!params.output_path.empty()) {
      params.output_path = ppforest2::io::json::ensure_extension(params.output_path);
    }

    // clang-format off
    #ifdef _OPENMP
    if (params.model.threads) {
      omp_set_num_threads(*params.model.threads);
    }
    #endif
    // clang-format on
    switch (params.subcommand) {
      case Subcommand::train: return run_train(params);
      case Subcommand::predict: return run_predict(params);
      case Subcommand::evaluate: return run_evaluate(params);
      case Subcommand::benchmark: return run_benchmark(params, argv[0]);
      case Subcommand::summarize: return run_summarize(params);
      case Subcommand::serve: return run_serve(params);

      default: fmt::print(stderr, "Error: No subcommand specified\n"); return 1;
    }
  } catch (ppforest2::UserError const& e) {
    fmt::print(stderr, "Error: {}\n", e.what());
    return 1;
  } catch (std::exception const& e) {
    // Backstop: anything not already translated into a UserError closer to
    // its source. Bad input must never escape as a raw exception — that
    // aborts via std::terminate with no usable message.
    fmt::print(stderr, "Error: {}\n", e.what());
    return 1;
  }
}