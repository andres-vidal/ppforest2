/**
 * @file Predict.cpp
 * @brief Predict subcommand handler.
 */
#include "cli/Predict.hpp"

#include <CLI/CLI.hpp>
#include <fmt/format.h>
#include <optional>
#include <vector>

#include "io/Color.hpp"
#include "io/IO.hpp"
#include "io/Output.hpp"
#include "io/Presentation.hpp"
#include "serialization/Json.hpp"
#include "stats/DataPacket.hpp"
#include "stats/Metrics.hpp"
#include "utils/UserError.hpp"

#include <nlohmann/json.hpp>

using namespace ppforest2;
using namespace ppforest2::types;
using namespace ppforest2::stats;
using namespace ppforest2::io::style;
using json = nlohmann::json;

namespace ppforest2::cli {
  void setup_predict(CLI::App& app, Params& params) {
    auto* sub = app.add_subcommand("predict", "Load a model and predict on new data");
    sub->add_option("-M,--model", params.model_path, "Saved model JSON file");
    sub->add_option("-d,--data", params.data_path, "CSV data to predict on");
    sub->add_option("-o,--output", params.output_path, "Save prediction results to JSON file");
    sub->add_flag("--no-metrics", params.no_metrics, "Omit error rate and confusion matrix from output");
    sub->add_flag("--no-proportions", params.no_proportions, "Omit vote proportions from output");

    // CLI-exclusive constraints (predict doesn't use config files)
    sub->get_option("--model")->required()->check(CLI::ExistingFile);
    sub->get_option("--data")->required()->check(CLI::ExistingFile);

    sub->callback([&]() { params.subcommand = Subcommand::predict; });
  }

  namespace {
    std::string model_display_name(json const& model_data) {
      return model_data.value("model_type", "tree") == "forest" ? "Random Forest" : "Decision Tree";
    }

  }

  int run_predict(Params const& params) {
    io::Output out(params.quiet);

    if (!params.output_path.empty()) {
      io::check_file_not_exists(params.output_path);
    }

    json const model_data = io::json::read_file(params.model_path, user_error);
    auto const exported   = model_data.get<serialization::Export<Model::Ptr>>();
    Mode const mode       = exported.spec->mode;

    DataPacket const data  = io::csv::read_sorted(params.data_path);
    auto const& model      = *exported.model;
    auto const predictions = model.predict(data.x);

    // `exported.groups` is the canonical training-label space — non-empty for
    // classification (validate_*_export enforces this), empty for regression.
    auto const& group_names = exported.groups;

    bool const has_labels      = data.y.size() > 0;
    bool const has_metrics     = has_labels && !params.no_metrics;
    bool const has_output_file = !params.output_path.empty();

    json result;
    if (has_output_file) {
      result["predictions"] = serialization::to_json(predictions, group_names);
    }

    if (has_output_file && is_classification(mode) && !params.no_proportions) {
      result["proportions"] = predict_proportions(model, data.x);
    }

    if (has_metrics) {
      Metrics const metrics = metrics_from_outcomes(predictions, data.y, mode);

      out.println("{}", emphasis("Prediction results for " + model_display_name(model_data)));
      out.newline();
      io::print_metrics_block(out, metrics, group_names);

      if (has_output_file) {
        result["metrics"] = serialization::to_json(metrics, group_names);
      }
    }

    if (has_metrics && !has_output_file) {
      out.println("{}", muted("Tip: use --output <file> to save individual predictions"));
    }

    if (has_output_file) {
      io::json::write_file(result, params.output_path, user_error);
      out.saved("Results", params.output_path);
    }

    return 0;
  }
}
