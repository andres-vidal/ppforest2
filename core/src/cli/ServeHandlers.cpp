/**
 * @file ServeHandlers.cpp
 * @brief Pure request → response functions for the `serve` subcommand.
 */
#include "cli/ServeHandlers.hpp"
#include "io/IO.hpp"
#include "serialization/Json.hpp"
#include "stats/Metrics.hpp"
#include "utils/UserError.hpp"

#include "app_css.hpp"
#include "predict_html.hpp"
#include "summary_html.hpp"
#include "summary_js.hpp"

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <inja/inja.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

using json = nlohmann::json;

namespace ppforest2::cli::serve {
  namespace {
    // Process-wide inja environment. Templates are static raw strings baked
    // into the binary; configuration is left at defaults (`{{ }}` for
    // interpolation, `{% %}` for control flow). No HTML auto-escaping —
    // string data is escaped at the data-construction layer instead.
    inja::Environment& templating_env() {
      static inja::Environment env;
      return env;
    }

    json bad_request(std::string message) {
      return json{{"error", std::move(message)}};
    }

    Response error_response(int status, std::string message) {
      return Response{status, bad_request(std::move(message)).dump()};
    }

    // Build a permutation `perm` of length n_features such that the model's
    // expected column j corresponds to the parsed CSV's column `perm[j]`.
    // Extra columns in @p provided (e.g. a response column carried over from
    // the training CSV) are silently ignored. Missing columns are an error.
    std::vector<int> build_permutation(types::Names const& expected, types::Names const& provided, std::string& err) {
      std::unordered_map<std::string, int> index;
      for (int i = 0; i < static_cast<int>(provided.size()); ++i) {
        index.emplace(provided[i], i);
      }

      std::vector<std::string> missing;
      std::vector<int> perm;
      perm.reserve(expected.size());

      for (auto const& name : expected) {
        auto it = index.find(name);
        if (it == index.end()) {
          missing.push_back(name);
        } else {
          perm.push_back(it->second);
        }
      }

      if (!missing.empty()) {
        err = fmt::format("Feature mismatch: missing [{}]", fmt::join(missing, ", "));
        return {};
      }

      return perm;
    }

    types::FeatureMatrix reorder_columns(types::FeatureMatrix const& src, std::vector<int> const& perm) {
      types::FeatureMatrix dst(src.rows(), static_cast<Eigen::Index>(perm.size()));
      for (int j = 0; j < static_cast<int>(perm.size()); ++j) {
        dst.col(j) = src.col(perm[j]);
      }
      return dst;
    }

    json proportions_to_json(types::FeatureMatrix const& probs) {
      json arr = json::array();
      for (Eigen::Index i = 0; i < probs.rows(); ++i) {
        json row = json::array();
        for (Eigen::Index j = 0; j < probs.cols(); ++j) {
          row.push_back(probs(i, j));
        }
        arr.push_back(std::move(row));
      }
      return arr;
    }
  }

  // PredictionStore ---------------------------------------------------------

  namespace {
    std::string generate_prediction_id() {
      // Random 64-bit hex IDs — opaque (don't leak ordering) and short enough
      // for shareable URLs (16 hex chars + leading "id=").
      thread_local std::random_device rd;
      uint64_t const hi = rd();
      uint64_t const lo = rd();
      uint64_t const x  = (hi << 32) | lo;
      return fmt::format("{:016x}", x);
    }
  }

  PredictionStore::PredictionStore(std::size_t capacity)
      : capacity(capacity > 0 ? capacity : DEFAULT_CAPACITY) {}

  std::string PredictionStore::put(PredictionResult result) {
    std::lock_guard<std::mutex> lock(mu);

    std::string id = generate_prediction_id();
    entries.emplace(id, std::move(result));
    order.push_back(id);

    while (order.size() > capacity) {
      entries.erase(order.front());
      order.pop_front();
    }
    return id;
  }

  std::optional<PredictionResult> PredictionStore::get(std::string const& id) const {
    std::lock_guard<std::mutex> lock(mu);
    auto const it = entries.find(id);
    if (it == entries.end()) {
      return std::nullopt;
    }
    return it->second;
  }

  Response handle_health(std::string const& version) {
    json body = {{"status", "ok"}, {"version", version}};
    return Response{200, body.dump()};
  }

  Response handle_summarize(LoadedModel const& loaded) {
    json copy = loaded.model_json;
    copy.erase("model");
    return Response{200, copy.dump(), "application/json"};
  }

  namespace {
    std::string html_escape(std::string const& s) {
      std::string out;
      out.reserve(s.size());
      for (char c : s) {
        switch (c) {
          case '&': out += "&amp;"; break;
          case '<': out += "&lt;"; break;
          case '>': out += "&gt;"; break;
          case '"': out += "&quot;"; break;
          case '\'': out += "&#39;"; break;
          default: out += c; break;
        }
      }
      return out;
    }

    std::string fmt_pct(double v) {
      return fmt::format("{:.1f}%", v * 100.0);
    }

    std::string fmt_num(double v) {
      return fmt::format("{:.4f}", v);
    }

    // Like fmt_num but with at most @p max_decimals digits and trailing
    // zeros trimmed: 0.5 → "0.5", 0.50001 → "0.5", 0.123 → "0.12", 1.0 → "1".
    // Used for config values (lambda, threshold, …) where 4-digit precision
    // is noisy.
    std::string fmt_compact(double v, int max_decimals) {
      std::string s  = fmt::format("{:.{}f}", v, max_decimals);
      auto const dot = s.find('.');
      if (dot != std::string::npos) {
        while (s.size() > 1 && s.back() == '0') {
          s.pop_back();
        }
        if (!s.empty() && s.back() == '.') {
          s.pop_back();
        }
      }
      return s;
    }

    // One-line model identifier used as the subtitle of /summary and /predict.
    // Examples:
    //   "Tree · Classification"
    //   "Forest of 100 trees · Classification"
    //   "Forest of 50 trees · Regression"
    std::string format_model_identifier(LoadedModel const& loaded) {
      json const& j                = loaded.model_json;
      bool const is_forest         = j.at("model_type").get<std::string>() == "forest";
      bool const is_classification = types::is_classification(loaded.mode);
      std::string const mode_label = is_classification ? "Classification" : "Regression";
      int const size               = j.at("config").at("size").get<int>();

      if (is_forest && size > 0) {
        return fmt::format("Forest of {} trees · {}", size, mode_label);
      }
      return fmt::format("{} · {}", is_forest ? "Forest" : "Tree", mode_label);
    }

    // Map raw snake_case strategy identifiers (as serialized in the config
    // JSON) to a display form: known abbreviations stay uppercase, everything
    // else is sentence-cased (first letter capitalized, underscores → spaces).
    std::string display_strategy_name(std::string const& raw) {
      if (raw == "pda") {
        return "PDA";
      }
      std::string out = raw;
      if (!out.empty()) {
        out[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(out[0])));
      }
      for (auto& c : out) {
        if (c == '_') {
          c = ' ';
        }
      }
      return out;
    }

    // Pretty-prints a config value: primitives become their string form;
    // strategy objects ({"name": "X", …params}) render as "X (k=v, …)"; arrays
    // join children with commas.
    std::string format_config_value(json const& v) {
      if (v.is_string()) {
        return v.get<std::string>();
      }
      if (v.is_boolean()) {
        return v.get<bool>() ? "true" : "false";
      }
      if (v.is_number_integer()) {
        return std::to_string(v.get<int64_t>());
      }
      if (v.is_number_float()) {
        // 2-decimal cap with trailing-zero trim — matches the precision the
        // user actually configures (e.g. lambda=0.5, threshold=0.01).
        return fmt_compact(v.get<double>(), 2);
      }
      if (v.is_null()) {
        return "—";
      }
      if (v.is_array()) {
        std::string out;
        bool first = true;
        for (auto const& item : v) {
          if (!first) {
            out += ", ";
          }
          out += format_config_value(item);
          first = false;
        }
        return out;
      }
      if (v.is_object()) {
        std::string const name = display_strategy_name(v.value("name", std::string("?")));
        std::string params;
        bool first = true;
        for (auto const& [key, val] : v.items()) {
          if (key == "name") {
            continue;
          }
          if (!first) {
            params += ", ";
          }
          params += key + "=" + format_config_value(val);
          first = false;
        }
        return params.empty() ? name : fmt::format("{} ({})", name, params);
      }
      return "?";
    }

    // ---- data builders for inja templates ---------------------------------

    json build_config_rows(json const& config) {
      static std::vector<std::pair<std::string, std::string>> const order = {
          {"pp", "Projection pursuit"},
          {"vars", "Variable selection"},
          {"cutpoint", "Cutpoint"},
          {"stop", "Stop rule"},
          {"grouping", "Grouping"},
          {"binarize", "Binarize"},
          {"leaf", "Leaf"},
          {"size", "Forest size"},
          {"seed", "Seed"},
          {"max_retries", "Max retries"},
      };

      json rows = json::array();
      for (auto const& [key, label] : order) {
        std::string value = format_config_value(config.at(key));
        if (key == "size" && value == "0") {
          value = "single tree";
        }
        rows.push_back({{"label", label}, {"value", html_escape(value)}});
      }
      return rows;
    }

    // Per-metrics-block rows (Accuracy/Error rate or MAE/MSE/R²) with an
    // optional label prefix ("OOB " for OOB blocks).
    json build_metric_rows(json const& metrics_json, bool is_classification, std::string const& prefix) {
      json rows = json::array();
      auto push = [&](std::string const& label, std::string const& value) {
        rows.push_back({{"label", prefix + label}, {"value", value}});
      };
      if (is_classification) {
        double const err = metrics_json.at("error_rate").get<double>();
        push("Accuracy", fmt_pct(1.0 - err));
        push("Error rate", fmt_pct(err));
      } else {
        push("MAE", fmt_num(metrics_json.at("mae").get<double>()));
        push("MSE", fmt_num(metrics_json.at("mse").get<double>()));
        push("R²", fmt_num(metrics_json.at("r_squared").get<double>()));
      }
      return rows;
    }

    // Summary-page metrics: stacks training + OOB blocks. Returns an empty
    // array when both blocks are null, so the template's
    // `{% if length(metrics) > 0 %}` correctly hides the section.
    json build_summary_metrics(json const& j, bool is_classification) {
      json rows = json::array();
      if (j.at("training_metrics").is_object()) {
        for (auto const& r : build_metric_rows(j.at("training_metrics"), is_classification, "")) {
          rows.push_back(r);
        }
      }
      if (j.at("oob_metrics").is_object()) {
        for (auto const& r : build_metric_rows(j.at("oob_metrics"), is_classification, "OOB ")) {
          rows.push_back(r);
        }
      }
      return rows;
    }

    json build_importance_rows(json const& j, types::Names const& feature_names) {
      json rows = json::array();
      if (!j.at("variable_importance").is_object()) {
        return rows;
      }
      auto const& values = j.at("variable_importance").at("permuted");

      std::vector<std::pair<int, double>> ranked;
      ranked.reserve(values.size());
      for (size_t i = 0; i < values.size() && i < feature_names.size(); ++i) {
        if (values[i].is_number()) {
          ranked.emplace_back(static_cast<int>(i), values[i].get<double>());
        }
      }
      if (ranked.empty()) {
        return rows;
      }
      std::sort(ranked.begin(), ranked.end(), [](auto const& a, auto const& b) { return a.second > b.second; });
      double const max_val = std::max(ranked.front().second, 1e-9);

      for (auto const& [i, v] : ranked) {
        int const width_pct = std::max(2, static_cast<int>(v / max_val * 100.0));
        rows.push_back({
            {"name", html_escape(feature_names[i])},
            {"width_pct", width_pct},
            {"value", fmt_num(v)},
        });
      }
      return rows;
    }

    // Builds {present, labels, rows} where each row carries
    // {label, cells: [{v, kind}], error}. `kind` is the colour-coding hint
    // — "correct" (diagonal), "wrong" (off-diagonal nonzero), "zero".
    json build_confusion_data(json const& metrics_json) {
      auto const& cm           = metrics_json.at("confusion_matrix");
      auto const& matrix       = cm.at("matrix");
      auto const& labels       = cm.at("labels");
      auto const& group_errors = cm.at("group_errors");
      int const n              = static_cast<int>(labels.size());

      json escaped_labels = json::array();
      for (auto const& l : labels) {
        escaped_labels.push_back(html_escape(l.get<std::string>()));
      }

      json rows = json::array();
      for (int i = 0; i < n; ++i) {
        json cells = json::array();
        for (int j = 0; j < n; ++j) {
          int const v            = matrix[i][j].get<int>();
          char const* const kind = (i == j) ? "correct" : (v > 0 ? "wrong" : "zero");
          cells.push_back({{"v", v}, {"kind", kind}});
        }
        rows.push_back({
            {"label", html_escape(labels[i].get<std::string>())},
            {"cells", cells},
            {"error", fmt_pct(group_errors[i].get<double>())},
        });
      }

      json out;
      out["present"] = true;
      out["labels"]  = escaped_labels;
      out["rows"]    = rows;
      return out;
    }
  }

  Response handle_summary_html(LoadedModel const& loaded) {
    json const& j                = loaded.model_json;
    bool const is_classification = types::is_classification(loaded.mode);

    json data;
    data["model_info"] = html_escape(format_model_identifier(loaded));
    // The `<style>...</style>` and `<script>...</script>` wrappers are
    // emitted from here (rather than living in the template) because some
    // HTML formatters re-format the contents of those tags as CSS/JS and
    // mangle the inja `{{ }}` placeholder.
    data["style_tag"]         = "<style>" + std::string(resources::APP_CSS) + "</style>";
    data["script_tag"]        = "<script>" + std::string(resources::SUMMARY_JS) + "</script>";
    data["version"]           = PPFOREST2_VERSION;
    data["obs"]               = j.at("meta").at("observations").get<int>();
    data["feat"]              = j.at("meta").at("features").get<int>();
    data["is_classification"] = is_classification;
    data["groups_count"]      = loaded.group_names.size();
    data["config"]            = build_config_rows(j.at("config"));
    data["metrics"]           = build_summary_metrics(j, is_classification);
    data["importance"]        = build_importance_rows(j, loaded.feature_names);

    std::string body = templating_env().render(resources::SUMMARY_HTML, data);
    return Response{200, std::move(body), "text/html; charset=utf-8"};
  }

  namespace {
    // The "response" column is whichever extra column in the request CSV
    // doesn't appear in the model's feature names. When multiple extras
    // exist, the last one wins — matching the CLI's convention where the
    // response is the rightmost column.
    std::optional<int> find_response_column(types::Names const& csv_columns, types::Names const& model_features) {
      std::optional<int> last_extra;
      for (int j = 0; j < static_cast<int>(csv_columns.size()); ++j) {
        if (std::find(model_features.begin(), model_features.end(), csv_columns[j]) == model_features.end()) {
          last_extra = j;
        }
      }
      return last_extra;
    }

    std::optional<types::OutcomeVector> parse_response_column(
        std::vector<std::vector<std::string>> const& raw_rows,
        int col,
        types::Mode mode,
        types::Names const& group_names
    ) {
      types::OutcomeVector y(static_cast<Eigen::Index>(raw_rows.size()));
      for (size_t i = 0; i < raw_rows.size(); ++i) {
        auto const& s = raw_rows[i][static_cast<size_t>(col)];
        if (types::is_classification(mode)) {
          auto const it = std::find(group_names.begin(), group_names.end(), s);
          if (it == group_names.end()) {
            return std::nullopt;
          }
          y(static_cast<Eigen::Index>(i)) = static_cast<types::Outcome>(std::distance(group_names.begin(), it));
        } else {
          try {
            y(static_cast<Eigen::Index>(i)) = std::stof(s);
          } catch (...) {
            return std::nullopt;
          }
        }
      }
      return y;
    }

    // Parse + validate + predict, shared by JSON and HTML predict handlers.
    // Returns either a populated result or a 4xx Response describing why the
    // request was rejected.
    std::variant<PredictionResult, Response> run_predict(LoadedModel const& loaded, std::string const& csv_body) {
      io::csv::FeatureSet parsed;
      try {
        parsed = io::csv::read_features_from_string(csv_body);
      } catch (UserError const& e) {
        return error_response(400, e.what());
      }

      std::string err;
      auto perm = build_permutation(loaded.feature_names, parsed.feature_names, err);
      if (perm.empty() && !loaded.feature_names.empty()) {
        return error_response(400, err);
      }

      types::FeatureMatrix const x = reorder_columns(parsed.x, perm);

      PredictionResult out;
      out.predictions = loaded.model->predict(x);
      if (types::is_classification(loaded.mode)) {
        out.proportions = predict_proportions(*loaded.model, x);
      }

      if (auto const response_col = find_response_column(parsed.feature_names, loaded.feature_names)) {
        if (auto y = parse_response_column(parsed.raw_rows, *response_col, loaded.mode, loaded.group_names)) {
          out.metrics = stats::metrics_from_outcomes(out.predictions, *y, loaded.mode);
        }
      }
      return out;
    }

    json build_predict_data(LoadedModel const& loaded, std::optional<PredictionResult> const& result) {
      json data;
      data["model_info"] = html_escape(format_model_identifier(loaded));
      // See note in handle_summary_html — the tags are emitted from C++ so
      // HTML formatters don't reformat the contents and break inja's
      // placeholder parsing.
      data["style_tag"]  = "<style>" + std::string(resources::APP_CSS) + "</style>";
      data["script_tag"] = "<script>" + std::string(resources::SUMMARY_JS) + "</script>";
      data["version"]    = PPFOREST2_VERSION;

      json result_data;
      if (!result) {
        result_data["present"]   = false;
        result_data["confusion"] = json{{"present", false}};
      } else {
        bool const is_classification = types::is_classification(loaded.mode);
        Eigen::Index const n         = result->predictions.size();

        json extra_headers = json::array();
        if (is_classification) {
          for (auto const& name : loaded.group_names) {
            extra_headers.push_back(html_escape(name));
          }
        }

        json rows = json::array();
        for (Eigen::Index i = 0; i < n; ++i) {
          std::string prediction;
          if (is_classification) {
            int const idx = static_cast<int>(std::lround(static_cast<double>(result->predictions(i))));
            prediction    = (idx >= 0 && static_cast<size_t>(idx) < loaded.group_names.size())
                                ? html_escape(loaded.group_names[static_cast<size_t>(idx)])
                                : std::to_string(idx);
          } else {
            prediction = fmt_num(static_cast<double>(result->predictions(i)));
          }

          json extra_cells = json::array();
          if (is_classification) {
            for (Eigen::Index k = 0; k < result->proportions.cols(); ++k) {
              extra_cells.push_back(fmt_pct(static_cast<double>(result->proportions(i, k))));
            }
          }
          rows.push_back({
              {"index", static_cast<int>(i + 1)},
              {"prediction", prediction},
              {"extra_cells", extra_cells},
          });
        }

        json metrics_arr = json::array();
        json confusion   = json{{"present", false}};
        if (result->metrics) {
          json const metrics_json = serialization::to_json(*result->metrics, loaded.group_names);
          metrics_arr             = build_metric_rows(metrics_json, is_classification, "");
          if (is_classification) {
            confusion = build_confusion_data(metrics_json);
          }
        }

        std::string const mode_label    = is_classification ? "Classification" : "Regression";
        std::string const count_caption = fmt::format("{} predictions · {}", n, mode_label);

        result_data["present"]       = true;
        result_data["count_caption"] = count_caption;
        result_data["extra_headers"] = extra_headers;
        result_data["rows"]          = rows;
        result_data["metrics"]       = metrics_arr;
        result_data["confusion"]     = confusion;
      }
      data["result"] = result_data;
      return data;
    }

    std::string render_predict_page(LoadedModel const& loaded, std::optional<PredictionResult> const& result) {
      json const data = build_predict_data(loaded, result);
      return templating_env().render(resources::PREDICT_HTML, data);
    }

    Response make_html_response(std::string body, std::string const& canonical_url) {
      Response r;
      r.status       = 200;
      r.body         = std::move(body);
      r.content_type = "text/html; charset=utf-8";
      if (!canonical_url.empty()) {
        r.headers["Content-Location"] = canonical_url;
      }
      return r;
    }

    std::string predict_url(std::string const& id) {
      return fmt::format("/predict?id={}", id);
    }
  }

  Response handle_predict(LoadedModel const& loaded, PredictionStore& store, std::string const& csv_body) {
    auto outcome = run_predict(loaded, csv_body);
    if (auto* err = std::get_if<Response>(&outcome)) {
      return std::move(*err);
    }
    auto& result          = std::get<PredictionResult>(outcome);
    std::string const id  = store.put(result); // copies into the store; result still usable below
    std::string const url = predict_url(id);

    json body;
    body["predictions"] = serialization::to_json(result.predictions, loaded.group_names);
    if (types::is_classification(loaded.mode)) {
      body["proportions"] = proportions_to_json(result.proportions);
    }
    if (result.metrics) {
      body["metrics"] = serialization::to_json(*result.metrics, loaded.group_names);
    }
    body["id"]  = id;
    body["url"] = url;

    Response r;
    r.status                      = 200;
    r.body                        = body.dump();
    r.content_type                = "application/json";
    r.headers["Content-Location"] = url;
    return r;
  }

  Response handle_predict_html(LoadedModel const& loaded, PredictionStore& store, std::string const& csv_body) {
    auto outcome = run_predict(loaded, csv_body);
    if (auto* err = std::get_if<Response>(&outcome)) {
      return std::move(*err);
    }
    auto result          = std::move(std::get<PredictionResult>(outcome));
    std::string const id = store.put(result); // copies into the store; result still usable below
    return make_html_response(render_predict_page(loaded, std::make_optional(std::move(result))), predict_url(id));
  }

  Response handle_predict_view(LoadedModel const& loaded, PredictionStore const& store, std::string const& id_query) {
    std::optional<PredictionResult> result;
    std::string canonical;
    if (!id_query.empty()) {
      result = store.get(id_query);
      if (result) {
        canonical = predict_url(id_query);
      }
      // Missing or evicted id: silently fall through to the form-only view.
    }
    return make_html_response(render_predict_page(loaded, result), canonical);
  }
}
