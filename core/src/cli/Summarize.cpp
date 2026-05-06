/**
 * @file Summarize.cpp
 * @brief Summarize subcommand handler: display saved model summary.
 */
#include "cli/Summarize.hpp"
#include "io/Presentation.hpp"
#include "io/Output.hpp"
#include "io/IO.hpp"
#include "serialization/Json.hpp"
#include "serialization/JsonOptional.hpp"
#include "utils/UserError.hpp"

#include <CLI/CLI.hpp>
#include <fmt/format.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using ppforest2::serialization::has_value;

namespace ppforest2::cli {
  void setup_summarize(CLI::App& app, Params& params) {
    auto* sub = app.add_subcommand("summarize", "Display a saved model summary");
    sub->add_option("-M,--model", params.model_path, "Saved model JSON file");
    sub->add_option("-d,--data", params.data_path, "CSV training data (recomputes metrics if provided)");

    // CLI-exclusive constraints (summarize doesn't use config files)
    sub->get_option("--model")->required()->check(CLI::ExistingFile);
    sub->get_option("--data")->check(CLI::ExistingFile);

    sub->callback([&]() { params.subcommand = Subcommand::summarize; });
  }

  int run_summarize(Params& params) {
    io::Output out(params.quiet);

    json model_data = io::json::read_file(params.model_path, user_error);

    // Recompute metrics if data is provided and metrics are absent
    // With the uniform `nullopt ↔ null` convention the keys are always
    // present after a fresh save; treat `null` the same as absent so a
    // `--no-metrics` model (writer skipped the computation entirely) and
    // an "all-null" model both recompute when data is provided.
    bool has_metrics = has_value(model_data, "training_metrics") || has_value(model_data, "variable_importance");

    if (!params.data_path.empty() && !has_metrics) {
      auto data         = io::csv::read_sorted(params.data_path);
      auto model_export = model_data.get<serialization::Export<Model::Ptr>>();
      model_export.compute_metrics(data.x, data.y);
      model_data = model_export.to_json();
    }

    io::print_summary(out, model_data);

    return 0;
  }
}
