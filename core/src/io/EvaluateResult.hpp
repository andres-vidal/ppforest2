/**
 * @file EvaluateResult.hpp
 * @brief Evaluation data types: per-iteration stats and run summaries.
 */
#pragma once

#include "stats/Stats.hpp"
#include "utils/Types.hpp"

#include <nlohmann/json.hpp>
#include <optional>
#include <string>

namespace ppforest2::io {
  /**
   * @brief Summary of an evaluation run.
   *
   * Contains data dimensions, aggregated timing/error metrics, and
   * optional memory usage.  Produced by ModelStats::summarize() on the
   * evaluate side, and deserialized from JSON on the benchmark side.
   */
  struct EvaluateResult {
    std::string data_path; ///< Data source path (empty for simulated data).

    types::Mode mode = types::Mode::Classification; ///< Training mode.
    int n            = 0;                           ///< Number of observations.
    int p            = 0;                           ///< Number of features.
    int g            = 0;                           ///< Number of groups (0 for regression).

    int size = 0;                ///< Forest size (number of trees).
    std::optional<int> n_vars;   ///< Variable count (integer mode).
    std::optional<float> p_vars; ///< Variable proportion.
    float train_ratio = 0.7F;    ///< Train/test split ratio.

    int runs             = 0;
    double mean_time_ms  = 0;
    double std_time_ms   = 0;
    double mean_tr_error = 0;
    double mean_te_error = 0;
    std::optional<long> peak_rss_bytes;
    std::optional<double> peak_rss_mb;

    EvaluateResult() = default;
    explicit EvaluateResult(nlohmann::json const& j);

    nlohmann::json to_json() const;
  };

  /**
   * @brief Per-iteration training statistics.
   *
   * Stores per-iteration timing and error vectors, plus the process-wide
   * peak RSS.  Provides mean/std accessors, an EvaluateResult summary,
   * and full JSON serialization (including a per-iteration breakdown).
   */
  struct ModelStats {
    std::string data_path;

    types::Mode mode = types::Mode::Classification; ///< Training mode.
    int n            = 0;                           ///< Number of observations.
    int p            = 0;                           ///< Number of features.
    int g            = 0;                           ///< Number of groups (0 for regression).

    int size = 0;
    std::optional<int> n_vars;
    std::optional<float> p_vars;
    float train_ratio = 0.7F;

    types::Vector<long long> tr_times;
    types::Vector<double> tr_error;
    types::Vector<double> te_error;
    long peak_rss_bytes = -1;

    double mean_time() const { return static_cast<double>(tr_times.mean()); }

    double mean_tr_error() const { return tr_error.mean(); }

    double mean_te_error() const { return te_error.mean(); }

    double std_time() const { return stats::sd(tr_times); }

    double std_tr_error() const { return stats::sd(tr_error); }

    double std_te_error() const { return stats::sd(te_error); }

    /** @brief Produce an EvaluateResult summary from per-iteration data. */
    EvaluateResult summarize() const;

    /** @brief Serialize to JSON including per-iteration breakdown. */
    nlohmann::json to_json() const;
  };
}
