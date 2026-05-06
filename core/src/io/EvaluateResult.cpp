/**
 * @file EvaluateResult.cpp
 * @brief EvaluateResult construction and serialization.
 */
#include "io/EvaluateResult.hpp"
#include "serialization/JsonOptional.hpp"

namespace ppforest2::io {
  nlohmann::json EvaluateResult::to_json() const {
    nlohmann::json j = {
        {"mode", types::to_string(mode)},
        {"n", n},
        {"p", p},
        {"g", g},
        {"size", size},
        {"train_ratio", train_ratio},
        {"runs", runs},
        {"mean_time_ms", mean_time_ms},
        {"std_time_ms", std_time_ms},
        {"mean_train_error", mean_tr_error},
        {"mean_test_error", mean_te_error},
    };

    if (!data_path.empty()) {
      j["data"] = data_path;
    }

    if (n_vars) {
      j["n_vars"] = *n_vars;
    }

    if (p_vars) {
      j["p_vars"] = *p_vars;
    }

    if (peak_rss_bytes) {
      j["peak_rss_bytes"] = *peak_rss_bytes;
      j["peak_rss_mb"]    = *peak_rss_mb;
    }

    return j;
  }

  EvaluateResult::EvaluateResult(nlohmann::json const& j)
      : data_path(j.value("data", ""))
      , mode(types::mode_from_string(j.value("mode", "classification")))
      , n(j.value("n", 0))
      , p(j.value("p", 0))
      , g(j.value("g", 0))
      , size(j.value("size", 0))
      , n_vars(j.value("n_vars", std::optional<int>{}))
      , p_vars(j.value("p_vars", std::optional<float>{}))
      , train_ratio(j.value("train_ratio", 0.7F))
      , runs(j.value("runs", 0))
      , mean_time_ms(j.value("mean_time_ms", 0.0))
      , std_time_ms(j.value("std_time_ms", 0.0))
      , mean_tr_error(j.value("mean_train_error", 0.0))
      , mean_te_error(j.value("mean_test_error", 0.0))
      , peak_rss_bytes(j.value("peak_rss_bytes", std::optional<long>{}))
      , peak_rss_mb(j.value("peak_rss_mb", std::optional<double>{})) {}

  EvaluateResult ModelStats::summarize() const {
    EvaluateResult r;
    r.data_path     = data_path;
    r.mode          = mode;
    r.n             = n;
    r.p             = p;
    r.g             = g;
    r.size          = size;
    r.n_vars        = n_vars;
    r.p_vars        = p_vars;
    r.train_ratio   = train_ratio;
    r.runs          = static_cast<int>(tr_times.size());
    r.mean_time_ms  = mean_time();
    r.std_time_ms   = std_time();
    r.mean_tr_error = mean_tr_error();
    r.mean_te_error = mean_te_error();

    if (peak_rss_bytes >= 0) {
      r.peak_rss_bytes = peak_rss_bytes;
      r.peak_rss_mb    = static_cast<double>(peak_rss_bytes) / (1024.0 * 1024.0);
    }

    return r;
  }

  nlohmann::json ModelStats::to_json() const {
    auto j = summarize().to_json();

    // Add fields only available from per-iteration data
    j["std_train_error"] = std_tr_error();
    j["std_test_error"]  = std_te_error();

    nlohmann::json iterations = nlohmann::json::array();
    for (int i = 0; i < tr_times.size(); ++i) {
      iterations.push_back({{"train_time_ms", tr_times[i]}, {"train_error", tr_error[i]}, {"test_error", te_error[i]}});
    }

    j["iterations"] = iterations;

    return j;
  }
}
