/**
 * @file Serve.cpp
 * @brief Serve subcommand handler: HTTP server around a loaded model.
 */
#include "cli/Serve.hpp"
#include "cli/ServeHandlers.hpp"
#include "io/Color.hpp"
#include "io/IO.hpp"
#include "io/Output.hpp"
#include "serialization/Json.hpp"
#include "utils/UserError.hpp"

#include <CLI/CLI.hpp>
#include <fmt/format.h>
#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <csignal>
#include <string>

using json = nlohmann::json;

namespace ppforest2::cli {
  namespace {
    std::atomic<httplib::Server*> g_server{nullptr};

    extern "C" void shutdown_signal_handler(int /*signum*/) {
      if (auto* s = g_server.load()) {
        s->stop();
      }
    }

    serve::LoadedModel load_model(std::string const& path) {
      json model_json        = io::json::read_file(path, user_error);
      auto exported          = model_json.get<serialization::Export<Model::Ptr>>();
      types::Mode const mode = exported.spec->mode;

      return serve::LoadedModel{
          exported.model,
          std::move(exported.groups),
          std::move(exported.feature_names),
          mode,
          std::move(model_json),
          exported.n_features,
      };
    }

    bool content_type_is_csv(std::string ct) {
      std::transform(ct.begin(), ct.end(), ct.begin(), [](unsigned char c) { return std::tolower(c); });
      return ct.empty() || ct.find("csv") != std::string::npos || ct.find("text/plain") != std::string::npos;
    }

    bool wants_html(httplib::Request const& req) {
      std::string const accept = req.get_header_value("Accept");
      return accept.find("text/html") != std::string::npos && accept.find("application/json") == std::string::npos;
    }

    void write_response(httplib::Response& res, serve::Response const& r) {
      res.status = r.status;
      res.set_content(r.body, r.content_type);
      for (auto const& [name, value] : r.headers) {
        res.set_header(name, value);
      }
    }
  }

  void setup_serve(CLI::App& app, Params& params) {
    auto& serve = params.serve;

    auto* sub = app.add_subcommand("serve", "Serve predictions over HTTP from a saved model");
    sub->add_option("-M,--model", params.model_path, "Saved model JSON file");
    sub->add_option("--host", serve.host, "Host to bind (default: 127.0.0.1)");
    sub->add_option("--port", serve.port, "Port to bind (default: 8080)");
    sub->add_option("--max-body-bytes", serve.max_body_bytes, "Maximum POST body size in bytes (default: 1048576)");

    sub->get_option("--model")->required()->check(CLI::ExistingFile);
    sub->get_option("--port")->check(CLI::Range(1, 65535));

    sub->callback([&]() { params.subcommand = Subcommand::serve; });
  }

  int run_serve(Params const& params) {
    io::Output out(params.quiet);

    serve::LoadedModel const loaded = load_model(params.model_path);
    serve::PredictionStore prediction_store;
    std::string const version = PPFOREST2_VERSION;

    httplib::Server server;
    server.set_payload_max_length(params.serve.max_body_bytes);
    server.set_read_timeout(30, 0);
    server.set_write_timeout(30, 0);
    server.set_keep_alive_timeout(30);

    server.Get("/health", [&](httplib::Request const&, httplib::Response& res) {
      write_response(res, serve::handle_health(version));
    });

    server.Get("/", [&](httplib::Request const& req, httplib::Response& res) {
      if (wants_html(req)) {
        write_response(res, serve::handle_summary_html(loaded));
      } else {
        write_response(res, serve::handle_summarize(loaded));
      }
    });

    server.Get("/predict", [&](httplib::Request const& req, httplib::Response& res) {
      write_response(res, serve::handle_predict_view(loaded, prediction_store, req.get_param_value("id")));
    });

    server.Post("/predict", [&](httplib::Request const& req, httplib::Response& res) {
      if (!content_type_is_csv(req.get_header_value("Content-Type"))) {
        res.status = 415;
        res.set_content(R"({"error":"Content-Type must be text/csv"})", "application/json");
        return;
      }

      if (wants_html(req)) {
        write_response(res, serve::handle_predict_html(loaded, prediction_store, req.body));
      } else {
        write_response(res, serve::handle_predict(loaded, prediction_store, req.body));
      }
    });

    server.set_logger([&](httplib::Request const& req, httplib::Response const& res) {
      out.println("{} {} -> {}", req.method, req.path, res.status);
    });

    g_server.store(&server);
    std::signal(SIGINT, shutdown_signal_handler);
    std::signal(SIGTERM, shutdown_signal_handler);

    if (params.serve.host == "0.0.0.0") {
      out.println("Warning: binding to 0.0.0.0 exposes the server externally without auth or TLS");
    }
    std::string const url = fmt::format("http://{}:{}", params.serve.host, params.serve.port);
    out.println("Serving {} on {}", params.model_path, io::style::link(url, io::style::info(url)));

    bool const ok = server.listen(params.serve.host, params.serve.port);
    g_server.store(nullptr);

    user_error(ok, fmt::format("Failed to bind to {}:{}", params.serve.host, params.serve.port));
    return 0;
  }
}
