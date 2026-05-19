/**
 * @file CLIOptions.hpp
 * @brief CLI argument parsing, validation, and configuration for ppforest2.
 *
 * Defines the Params struct and declares functions to parse,
 * validate, and initialize runtime parameters.
 */
#pragma once

#include "cli/ModelParams.hpp"
#include "cli/EvaluateParams.hpp"
#include "cli/BenchmarkParams.hpp"
#include "cli/ServeParams.hpp"
#include "io/Output.hpp"

#include <nlohmann/json.hpp>
#include <string>

/**
 * @brief Command-line interface: argument parsing, subcommands, and
 *        benchmark/evaluation orchestration.
 */
namespace ppforest2::cli {
  /** @brief Available CLI subcommands. */
  enum class Subcommand : uint8_t { none, train, predict, evaluate, benchmark, summarize, serve };

  /**
   * @brief All CLI options and runtime parameters.
   *
   * Fields with -1 or empty defaults are sentinel values meaning
   * "not set by the user" and will be resolved by init_params().
   */
  struct Params {
    Subcommand subcommand = Subcommand::none;

    ModelParams model;
    SimulateParams simulation;
    EvaluateParams evaluate;
    BenchmarkParams benchmark;
    ServeParams serve;

    std::string data_path;
    std::string save_path = "model.json";
    std::string model_path;
    std::string output_path;

    bool quiet          = false;
    bool no_save        = false;
    bool no_metrics     = false;
    bool no_color       = false;
    bool no_proportions = false;

    Params() = default;

    /** @brief Construct from a JSON config object. */
    explicit Params(nlohmann::json const& config);

    /**
     * @brief Resolve intermediate representations into final form.
     *
     * Parses strategy input strings into config JSON, resolves p_vars
     * input, parses simulate format, and runs validation. Called after
     * all sources (config file + CLI) have populated the fields.
     */
    void resolve();

    /** @brief Generate a random seed if none was set. */
    void resolve_seed();

    /**
     * @brief Load the training data: from the `--data` CSV when set,
     *        or simulated from `--simulate` params otherwise.
     *
     * Side effect: when `model.mode_input` is empty, infers it — from the
     * CSV's y column shape (regression if `group_names` is empty) or
     * defaults to "classification" for simulated data.
     */
    stats::DataPacket read_data(stats::RNG& rng);

    /**
     * @brief Fill in runtime defaults (seed, threads, vars, strategy configs).
     *
     * @param total_vars Total number of feature columns.
     */
    void resolve_defaults(unsigned int total_vars);

    /** @brief Serialize to a JSON config (round-trips with the JSON constructor). */
    nlohmann::json to_json() const;
  };

  /**
   * @brief Warn the user about parameters that are ignored for single-tree training.
   */
  void warn_unused_params(io::Output& out, Params const& params);

  /**
   * @brief Parse command-line arguments into a Params struct.
   *
   * @param argc Argument count from main().
   * @param argv Argument vector from main().
   * @return A populated Params struct.
   */
  Params parse_args(int argc, char* argv[]);
}
