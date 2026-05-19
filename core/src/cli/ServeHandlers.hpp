/**
 * @file ServeHandlers.hpp
 * @brief Pure request → response functions for the `serve` subcommand.
 *
 * Handlers are pure functions of (cached model snapshot, request body)
 * so they can be unit-tested without bringing up an HTTP server.
 */
#pragma once

#include "models/Model.hpp"
#include "stats/Metrics.hpp"
#include "utils/Types.hpp"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <deque>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

namespace ppforest2::cli::serve {
  /** @brief Immutable model snapshot captured once at server startup. */
  struct LoadedModel {
    Model::Ptr model;
    types::Names group_names;
    types::Names feature_names;
    types::Mode mode = types::Mode::Classification;
    nlohmann::json model_json;
  };

  /** @brief Status, body, MIME content type, and extra response headers. */
  struct Response {
    int status = 200;
    std::string body;
    std::string content_type = "application/json";
    std::map<std::string, std::string> headers;
  };

  /** @brief Result of running a prediction request — kept around for the
   *         predictions store so the server can serve cached HTML on
   *         shareable URLs. */
  struct PredictionResult {
    types::OutcomeVector predictions;
    types::FeatureMatrix proportions;      // populated for classification only
    std::optional<stats::Metrics> metrics; // present when the request carried a response column
  };

  /** @brief In-memory bounded store of recent prediction results, keyed by
   *         opaque IDs that survive a page refresh. Capacity-limited LRU;
   *         evicts the oldest entry once capacity is reached.
   *
   * Predictions are kept only in memory; restarting the server clears them.
   * IDs are hex-encoded 64-bit random integers — opaque enough that they
   * don't reveal request counts.
   */
  class PredictionStore {
  public:
    static constexpr std::size_t DEFAULT_CAPACITY = 32;

    explicit PredictionStore(std::size_t capacity = DEFAULT_CAPACITY);

    /** @brief Store @p result and return its newly-generated id. */
    std::string put(PredictionResult result);

    /** @brief Look up @p id; returns nullopt if not in the store (evicted or never existed). */
    std::optional<PredictionResult> get(std::string const& id) const;

  private:
    mutable std::mutex mu;
    std::size_t capacity;
    std::deque<std::string> order;
    std::unordered_map<std::string, PredictionResult> entries;
  };

  /** @brief `GET /health` → 200 with status + version. */
  Response handle_health(std::string const& version);

  /**
   * @brief `GET /` (Accept: application/json) → 200 with the loaded
   *        model JSON, minus the heavy `model` field.
   */
  Response handle_summarize(LoadedModel const& loaded);

  /**
   * @brief `GET /` (Accept: text/html) → 200 with a self-contained HTML page
   *        rendering meta, training metrics, and variable importance.
   */
  Response handle_summary_html(LoadedModel const& loaded);

  /**
   * @brief `POST /predict` (Accept: application/json) → 200 with
   *        `{predictions, [proportions], [metrics], id}`.
   *
   * Stores the result in @p store so the corresponding GET endpoint can
   * return the same data later. The `id` field in the body and the
   * `Content-Location: /predict?id=<id>` header point to the canonical URL.
   *
   * @return 400 on malformed CSV / feature-name mismatch.
   */
  Response handle_predict(LoadedModel const& loaded, PredictionStore& store, std::string const& csv_body);

  /**
   * @brief `POST /predict` (Accept: text/html) → 200 with the predict page
   *        rendered against the new result. `Content-Location` header points
   *        to the canonical /predict?id=<id> URL so the browser can update
   *        its address bar and share/refresh works.
   */
  Response handle_predict_html(LoadedModel const& loaded, PredictionStore& store, std::string const& csv_body);

  /**
   * @brief `GET /predict` → 200 with the predict page (always showing the
   *        upload form). When @p id_query refers to an entry still in the
   *        store, the results section is also populated.
   */
  Response handle_predict_view(LoadedModel const& loaded, PredictionStore const& store, std::string const& id_query);
}
