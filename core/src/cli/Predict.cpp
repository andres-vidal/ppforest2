/**
 * @file Predict.cpp
 * @brief Predict subcommand handler.
 */
#include "cli/Predict.hpp"
#include "ppforest2.hpp"

#include <CLI/CLI.hpp>
#include <fmt/format.h>
#include <fstream>
#include <vector>

#include "stats/DataPacket.hpp"
#include "stats/ConfusionMatrix.hpp"
#include "io/Presentation.hpp"
#include "io/Color.hpp"
#include "io/Output.hpp"
#include "io/IO.hpp"
#include "serialization/Json.hpp"
#include "utils/UserError.hpp"

#include <nlohmann/json.hpp>

using namespace ppforest2;
using namespace ppforest2::types;
using namespace ppforest2::stats;
using namespace ppforest2::io::style;
using json = nlohmann::json;

namespace ppforest2::cli {
  CLI::App * setup_predict(CLI::App& app, CLIOptions& params) {
    auto sub = app.add_subcommand("predict", "Load a model and predict on new data");
    sub->add_option("-M,--model", params.model_path, "Saved model JSON file")
    ->required()
    ->check(CLI::ExistingFile);
    sub->add_option("-d,--data", params.data_path, "CSV data to predict on")
    ->required()
    ->check(CLI::ExistingFile);
    sub->add_option("-o,--output", params.output_path, "Save prediction results to JSON file");
    sub->add_flag("--no-metrics", params.no_metrics, "Omit error rate and confusion matrix from output");
    sub->add_flag("--no-proportions", params.no_proportions, "Omit vote proportions from output");
    return sub;
  }
}

namespace ppforest2::cli {
namespace {
  json proportions_to_json(const FeatureMatrix& proportions) {
    std::vector<std::vector<float>> rows;
    rows.reserve(proportions.rows());

    for (Eigen::Index i = 0; i < proportions.rows(); ++i) {
      std::vector<float> row(proportions.cols());

      for (Eigen::Index j = 0; j < proportions.cols(); ++j) {
        row[j] = proportions(i, j);
      }

      rows.push_back(std::move(row));
    }

    return rows;
  }

  struct ProportionsVisitor : Model::Visitor {
    const FeatureMatrix& data;
    FeatureMatrix proportions;
    bool has_proportions = false;

    explicit ProportionsVisitor(const FeatureMatrix& data) : data(data) {
    }

    void visit(const Tree& tree) override {
      has_proportions = true;
      proportions     = tree.predict(data, Proportions{});
    }

    void visit(const Forest& forest) override {
      has_proportions = true;
      proportions     = forest.predict(data, Proportions{});
    }
  };

  json build_predict_result(
    const ResponseVector&           predictions,
    const DataPacket&               data,
    const Model&                    model,
    const std::vector<std::string>& group_names,
    bool no_metrics,
    bool no_proportions) {
    json result;

    if (group_names.empty()) {
      std::vector<int> pred_vec(predictions.data(), predictions.data() + predictions.size());
      result["predictions"] = pred_vec;
    } else {
      result["predictions"] = serialization::to_labels(predictions, group_names);
    }

    bool has_labels   = data.y.size() > 0;
    bool show_metrics = has_labels && !no_metrics;

    if (show_metrics) {
      ConfusionMatrix cm(predictions, data.y);
      result["error_rate"]       = cm.error();
      result["confusion_matrix"] = group_names.empty()
        ? serialization::to_json(cm)
        : serialization::to_json(cm, group_names);
    }

    if (!no_proportions) {
      ProportionsVisitor visitor(data.x);
      model.accept(visitor);

      if (visitor.has_proportions) {
        result["proportions"] = proportions_to_json(visitor.proportions);
      }
    }

    return result;
  }
}

  int run_predict(CLIOptions& params) {
    io::Output out(params.quiet);

    // Validate output path before doing work
    if (!params.output_path.empty()) {
      io::check_file_not_exists(params.output_path);
    }

    DataPacket data = [&]() {
      try {
        return io::csv::read_sorted(params.data_path);
      } catch (const ppforest2::UserError& e) {
        fmt::print(stderr, "{} {}\n", error("Error:"), e.what());
        fmt::print(stderr, "File: {}\n", params.data_path);
        std::exit(1);
      } catch (const std::exception& e) {
        fmt::print(stderr, "{} reading CSV file: {}\n", error("Error:"), e.what());
        std::exit(1);
      }
    }();

    json model_data  = io::json::read_file(params.model_path);
    auto model       = serialization::model_from_json(model_data);
    auto predictions = model->predict(data.x);

    bool has_labels   = data.y.size() > 0;
    bool show_metrics = has_labels && !params.no_metrics;

    // Resolve group names: prefer CSV data, fall back to saved model metadata.
    std::vector<std::string> group_names = data.group_names;

    if (group_names.empty() && model_data.contains("meta") && model_data["meta"].contains("groups")) {
      group_names = model_data["meta"]["groups"].get<std::vector<std::string>>();
    }

    // Terminal output
    if (show_metrics) {
      std::string model_type = model_data.value("model_type", "tree") == "forest"
        ? "Random Forest" : "Decision Tree";

      out.println("{}", emphasis("Prediction results for " + model_type));
      out.newline();

      ConfusionMatrix cm(predictions, data.y);
      out.println("{} {}", emphasis("Error:"), fmt::format("{:.2f}%", cm.error() * 100));
      print_confusion_matrix(out, cm, "Confusion Matrix", group_names);
    }

    // Hint about --output when not used
    if (show_metrics && params.output_path.empty()) {
      out.println("{}", muted("Tip: use --output <file> to save individual predictions"));
    }

    // Save results to file if requested
    if (!params.output_path.empty()) {
      json file_result = build_predict_result(predictions, data, *model, group_names, params.no_metrics, params.no_proportions);
      io::json::write_file(file_result, params.output_path);
      out.saved("Results", params.output_path);
    }

    return 0;
  }
}
