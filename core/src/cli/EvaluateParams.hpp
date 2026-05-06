/**
 * @file EvaluateParams.hpp
 * @brief Evaluate-specific CLI parameters: simulation, convergence, and evaluation options.
 */
#pragma once

#include "stats/Simulation.hpp"

#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

namespace ppforest2::cli {
  /** @brief Simulation data source options. */
  struct SimulateParams {
    SimulateParams() = default;

    std::string format;
    int rows     = 1000;
    int cols     = 10;
    int n_groups = 2;

    stats::simulation::params::Classification classification;
    stats::simulation::params::Regression regression;

    /** @brief Construct from a JSON config object. */
    explicit SimulateParams(nlohmann::json const& config);

    /** @brief Parse the NxPxG format string into rows, cols, n_groups. */
    void resolve_format();

    /**
     * @brief Generate a simulated dataset for the given mode.
     *
     * Dispatches to the classification (group-shifted normals) or
     * regression (linear model) generator based on @p mode. Any failure
     * surfaces as a `UserError` — simulation parameters come from the
     * user, not from a programmer-controlled invariant.
     */
    stats::DataPacket simulate(types::Mode mode, stats::RNG& rng) const;
  };

  /** @brief Evaluate and convergence options (shared by evaluate + benchmark). */
  struct EvaluateParams {
    EvaluateParams() = default;

    std::optional<float> train_ratio;
    std::optional<int> iterations; ///< Fixed count (disables convergence when set > 0).
    int warmup = 0;                ///< Warmup iterations discarded before measuring.
    /**
     * @brief Use the user-provided seed unchanged for every iteration.
     *
     * Default (`false`): each iteration draws a fresh seed from `rng` so
     * models genuinely differ across iterations. `std_*_error` reflects
     * real algorithmic variance; `std_time_ms` mixes system noise with
     * model-shape variance.
     *
     * `true`: every iteration trains the *same* model. `std_*_error` is
     * tautologically zero; `std_time_ms` reflects pure timing noise (cache,
     * scheduling, allocator jitter). Useful when benchmarking the timing
     * stability of a single configuration.
     */
    bool fixed_seed = false;
    std::string export_path;

    /** @brief Whether adaptive convergence stopping is active (disabled when iterations is set). */
    bool convergence_enabled() const { return !iterations || *iterations <= 0; }

    /** @brief Convergence control. */
    struct Convergence {
      std::optional<float> cv;   ///< CV target (e.g. 0.05 = stop when std < 5% of mean).
      std::optional<int> min;    ///< Minimum iterations before checking convergence.
      std::optional<int> window; ///< Consecutive checks below threshold before stopping.
      std::optional<int> max;    ///< Hard upper bound on iterations.
    } convergence;

    /** @brief Validate evaluate-related fields in a JSON config. */
    static void validate(nlohmann::json const& config, std::vector<std::string>& errors);

    /** @brief Construct from a JSON config object. */
    explicit EvaluateParams(nlohmann::json const& config);

    /** @brief Fill unset optional fields with their default values. */
    void resolve_defaults();

    /** @brief Serialize to JSON config. */
    nlohmann::json to_json() const;

    /** @brief JSON with only the fields explicitly set (for scenario overrides). */
    nlohmann::json overrides() const;
  };
}
