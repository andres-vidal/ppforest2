/**
 * @file Presentation.cpp
 * @brief Model statistics, variable importance serialization, and
 *        formatted terminal display for confusion matrices and VI tables.
 */
#include "io/Presentation.hpp"
#include "io/Table.hpp"
#include "models/strategies/binarize/Binarization.hpp"
#include "models/strategies/cutpoint/Cutpoint.hpp"
#include "models/strategies/leaf/LeafStrategy.hpp"
#include "models/strategies/grouping/Grouping.hpp"
#include "models/strategies/pp/ProjectionPursuit.hpp"
#include "models/strategies/stop/StopRule.hpp"
#include "models/strategies/vars/VariableSelection.hpp"
#include "serialization/Json.hpp"
#include "serialization/JsonOptional.hpp"
#include "utils/RangeVector.hpp"

#include <fmt/format.h>
#include <algorithm>
#include <cmath>
#include <vector>

using namespace ppforest2::serialization;

namespace ppforest2::io {
  void print_results(Output& out, ModelStats const& stats) {
    using namespace style;
    using namespace layout;

    out.println("{}", emphasis("Evaluation results"));
    out.newline();

    std::vector<Column> columns = {
        {"Runs", 5, Align::left},
        {"Time (ms)", 18, Align::right},
        {"Train Err", 10, Align::right},
        {"Test Err", 10, Align::right},
    };

    if (stats.peak_rss_bytes >= 0) {
      columns.push_back({"Peak RSS", 10, Align::right});
    }

    Row const header = header_labels(columns);

    out.println("{}", format_row(columns, header));
    out.println("{}", muted(format_separator(columns)));

    std::string const time_str = fmt::format("{:.2f} +/- {:.2f}", stats.mean_time(), stats.std_time());
    std::string tr_err;
    std::string te_err;

    if (stats.mode == types::Mode::Classification) {
      tr_err = fmt::format("{:.2f}%", stats.mean_tr_error() * 100);
      te_err = fmt::format("{:.2f}%", stats.mean_te_error() * 100);
    } else {
      tr_err = fmt::format("{:.4f}", stats.mean_tr_error());
      te_err = fmt::format("{:.4f}", stats.mean_te_error());
    }

    Row cells = {
        fmt::format("{}", stats.tr_times.size()),
        time_str,
        tr_err,
        te_err,
    };

    if (stats.peak_rss_bytes >= 0) {
      double mb = static_cast<double>(stats.peak_rss_bytes) / (1024.0 * 1024.0);
      cells.push_back(fmt::format("{:.1f} MB", mb));
    }

    out.println("{}", format_row(columns, cells));
    out.newline();
  }

  void print_variable_importance(
      Output& out, VariableImportance const& vi, types::Names const& feature_names, int max_rows
  ) {
    using namespace style;
    using namespace layout;

    auto const& vi1   = vi.permuted;
    auto const& vi2   = vi.projections;
    auto const& vi3   = vi.weighted_projections;
    auto const& scale = vi.scale;

    int const p = static_cast<int>(vi2.size());

    bool const has_names = static_cast<int>(feature_names.size()) == p;

    std::vector<int> order = utils::range_vector(p);
    std::stable_sort(order.begin(), order.end(), [&vi2](int a, int b) { return std::isgreater(vi2(a), vi2(b)); });

    bool const show_vi1 = vi1.size() == vi2.size();
    bool const show_vi3 = vi3.size() == vi2.size();
    int const rows      = (max_rows > 0 && p > max_rows) ? max_rows : p;

    // Compute variable column width based on longest name.
    int var_width = 10;

    if (has_names) {
      for (int rank = 0; rank < rows; ++rank) {
        int const j   = order[static_cast<std::size_t>(rank)];
        int const len = static_cast<int>(feature_names[static_cast<std::size_t>(j)].size());


        var_width = std::max(var_width, len + 1);
      }
    }

    out.println("{}", emphasis("Variable Importance:"));
    out.newline();

    // Build columns conditionally
    std::vector<Column> columns = {
        {"Variable", var_width, Align::left},
        {"", 10, Align::right},
        {"Projection", 12, Align::right},
    };

    if (show_vi3) {
      columns.push_back({"Weighted", 12, Align::right});
    }

    if (show_vi1) {
      columns.push_back({"Permuted", 12, Align::right});
    }

    // Header
    Row header = header_labels(columns);
    header[0]  = emphasis(header[0]);
    header[1]  = muted(fmt::format("{}", "\xcf\x83"));
    header[2]  = emphasis(header[2]);

    if (show_vi3) {
      header[3] = emphasis(header[3]);
    }
    if (show_vi1) {
      header[show_vi3 ? 4 : 3] = emphasis(header[show_vi3 ? 4 : 3]);
    }
    out.println("{}", format_row(columns, header));
    out.println("{}", muted(format_separator(columns)));

    // Data rows
    for (int rank = 0; rank < rows; ++rank) {
      int const j         = order[static_cast<std::size_t>(rank)];
      std::string const v = has_names ? feature_names[static_cast<std::size_t>(j)] : fmt::format("x{}", j + 1);

      Row cells = {
          v,
          fmt::format("{:.4f}", scale(j)),
          fmt::format("{:.6f}", vi2(j)),
      };

      if (show_vi3) {
        cells.push_back(fmt::format("{:.6f}", vi3(j)));
      }

      if (show_vi1) {
        cells.push_back(fmt::format("{:.6f}", vi1(j)));
      }

      out.println("{}", format_row(columns, cells));
    }

    if (rows < p) {
      out.println("{}", muted(fmt::format("... {} more variables not shown", p - rows)));
    }

    bool const all_ones = (scale.array() - types::Feature(1)).abs().maxCoeff() < types::Feature(1e-6);

    if (!all_ones) {
      out.newline();
      out.newline();
      out.println(
          "{} Variable importance was calculated using scaled coefficients (|a_j| * \u03c3_j).",
          emphasis(warning("Note:"))
      );
      out.println("Variable contributions can only be theoretically interpreted as such");
      out.println("if the model was trained on scaled data. Scaling also changes the");
      out.println("projection-pursuit optimization, which may affect the resulting tree.");
    }

    out.newline();
  }

  void print_confusion_matrix(
      Output& out, stats::ConfusionMatrix const& cm, std::string const& title, types::Names const& group_names
  ) {
    using namespace style;

    auto group_err = cm.group_errors();

    bool const has_names = !group_names.empty();

    // Compute column width: max of group name lengths and default width (5).
    int col_width = 5;

    if (has_names) {
      for (auto const& [label, idx] : cm.label_index) {
        int const name_len = static_cast<int>(group_names[static_cast<std::size_t>(label)].size());

        col_width = std::max(name_len + 1, col_width);
      }
    }

    // Compute row label width.
    int row_label_width = 4;

    if (has_names) {
      for (auto const& [label, idx] : cm.label_index) {
        int const name_len = static_cast<int>(group_names[static_cast<std::size_t>(label)].size());

        row_label_width = std::max(name_len, row_label_width);
      }
    }

    out.println("{}", emphasis(title + ":"));
    out.newline();

    // Header row: group labels
    std::string header(static_cast<std::size_t>(row_label_width), ' ');

    for (auto const& [label, idx] : cm.label_index) {
      std::string name = has_names ? group_names[static_cast<std::size_t>(label)] : std::to_string(label);
      header += fmt::format("{:>{}}", name, col_width);
    }

    header += "  Error";
    out.println("{}", header);

    // Data rows
    for (auto const& [label, row_idx] : cm.label_index) {
      std::string row_label = has_names ? group_names[static_cast<std::size_t>(label)] : std::to_string(label);
      std::string row       = fmt::format("{:>{}}", row_label, row_label_width);

      for (auto const& [col_label, col_idx] : cm.label_index) {
        int val                = cm.values(row_idx, col_idx);
        std::string const cell = fmt::format("{:>{}}", val, col_width - 1);

        if (row_idx == col_idx) {
          row += " " + success(cell);
        } else if (val > 0) {
          row += " " + error(cell);
        } else {
          row += " " + muted(cell);
        }
      }

      row += fmt::format("  {:.1f}%", group_err[row_idx] * 100);
      out.println("{}", row);
    }

    out.newline();
  }

  void print_regression_metrics(Output& out, stats::RegressionMetrics const& rm, std::string const& title) {
    using namespace style;
    using namespace layout;

    out.println("{}", emphasis(title + ":"));
    out.newline();

    std::vector<Column> columns = {
        {"Metric", 18, Align::left},
        {"Value", 18, Align::right},
    };

    Row header = header_labels(columns);
    out.println("{}", format_row(columns, header));
    out.println("{}", muted(format_separator(columns)));

    out.println("{}", format_row(columns, {"MSE", fmt::format("{:.6f}", rm.mse)}));
    out.println("{}", format_row(columns, {"MAE", fmt::format("{:.6f}", rm.mae)}));
    out.println("{}", format_row(columns, {"R\xc2\xb2", fmt::format("{:.6f}", rm.r_squared)}));

    out.newline();
  }

  namespace {
    // Empty label produces "Error:"; "Training" produces "Training Error:".
    std::string prefixed(std::string const& label, std::string const& title) {
      return label.empty() ? title : label + " " + title;
    }
  }

  void print_metrics_block(
      Output& out, stats::ClassificationMetrics const& cm, std::string const& label, types::Names const& group_names
  ) {
    using namespace style;
    out.println("{} {}", emphasis(prefixed(label, "Error:")), fmt::format("{:.2f}%", cm.error_rate() * 100));
    print_confusion_matrix(out, cm.confusion_matrix, prefixed(label, "Confusion Matrix"), group_names);
  }

  void print_metrics_block(
      Output& out, stats::RegressionMetrics const& rm, std::string const& label, types::Names const& /*group_names*/
  ) {
    using namespace style;
    out.println("{} {}", emphasis(prefixed(label, "MSE:")), fmt::format("{:.6f}", rm.mse));
    print_regression_metrics(out, rm, prefixed(label, "Regression Metrics"));
  }

  void print_metrics_block(
      Output& out, stats::Metrics const& metrics, std::string const& label, types::Names const& group_names
  ) {
    std::visit([&](auto const& m) { print_metrics_block(out, m, label, group_names); }, metrics);
  }

  void print_metrics_block(Output& out, stats::ClassificationMetrics const& m, types::Names const& group_names) {
    print_metrics_block(out, m, "", group_names);
  }

  void print_metrics_block(Output& out, stats::RegressionMetrics const& m, types::Names const& group_names) {
    print_metrics_block(out, m, "", group_names);
  }

  void print_metrics_block(Output& out, stats::Metrics const& m, types::Names const& group_names) {
    print_metrics_block(out, m, "", group_names);
  }

  namespace {
    void print_metrics_from_json(
        Output& out,
        nlohmann::json const& model_data,
        std::string const& key,
        std::string const& label,
        types::Mode mode,
        types::Names const& group_names
    ) {
      if (!serialization::has_value(model_data, key)) {
        return;
      }
      print_metrics_block(out, serialization::metrics_from_json(model_data[key], mode), label, group_names);
    }
  }

  namespace {
    std::string json_value_to_string(nlohmann::json const& v) {
      if (v.is_string()) {
        return v.get<std::string>();
      }

      if (v.is_number_integer()) {
        return std::to_string(v.get<int>());
      }

      if (v.is_number_float()) {
        return fmt::format("{}", v.get<double>());
      }

      if (v.is_boolean()) {
        return v.get<bool>() ? "true" : "false";
      }

      return v.dump();
    }
  }

  void print_configuration(Output& out, nlohmann::json const& config, ConfigDisplayHints const& hints) {
    using namespace style;
    using namespace layout;

    auto dtag = [&](bool is_default) -> std::string {
      return is_default ? " " + muted("(default)") : "";
    };

    int size = config.value("size", 0);

    std::string model_type = size > 0 ? "Random Forest of Projection-Pursuit Oblique Decision Trees"
                                      : "Projection-Pursuit Oblique Decision Tree";

    out.println("{}", emphasis(model_type));
    out.newline();

    std::vector<Column> columns = {
        {"Parameter", 18, Align::left},
        {"Value", 30, Align::left},
    };

    Row header = header_labels(columns);
    out.println("{}", format_row(columns, header));
    out.println("{}", muted(format_separator(columns)));

    // Strategy sections: pp, vars, cutpoint
    // Resolve display names by instantiating strategies from JSON via the registry.
    auto strategy_display_name = [](std::string const& key, nlohmann::json const& section) -> std::string {
      try {
        if (key == "pp") {
          return pp::ProjectionPursuit::from_json(section)->display_name();
        }

        if (key == "vars") {
          return vars::VariableSelection::from_json(section)->display_name();
        }

        if (key == "cutpoint") {
          return cutpoint::Cutpoint::from_json(section)->display_name();
        }

        if (key == "stop") {
          return stop::StopRule::from_json(section)->display_name();
        }

        if (key == "binarize") {
          return binarize::Binarization::from_json(section)->display_name();
        }

        if (key == "grouping") {
          return grouping::Grouping::from_json(section)->display_name();
        }

        if (key == "leaf") {
          return leaf::LeafStrategy::from_json(section)->display_name();
        }
      } catch (std::exception const& e) {
        fmt::print(stderr, "Warning: could not resolve display name for strategy '{}': {}\n", key, e.what());
      }
      return section.value("name", key);
    };

    for (auto const& key : {"pp", "vars", "cutpoint"}) {
      if (!config.contains(key)) {
        continue;
      }

      auto const& section       = config[key];
      std::string const display = strategy_display_name(key, section);
      out.println("{}", format_row(columns, {std::string(key) + " method", display}));

      for (auto const& [k, v] : section.items()) {
        if (k == "name") {
          continue;
        }

        std::string value = json_value_to_string(v);

        if (k == "count" && hints.vars_percent >= 0) {
          value += fmt::format(" ({:.0f}%)", hints.vars_percent);
        }

        if (k == "count" && hints.default_vars) {
          value += dtag(true);
        }

        out.println("{}", format_row(columns, {k, value}));
      }
    }

    // Top-level parameters
    if (size > 0) {
      out.println("{}", format_row(columns, {"trees", std::to_string(size)}));
      out.println(
          "{}",
          format_row(columns, {"threads", fmt::format("{}{}", config.value("threads", 1), dtag(hints.default_threads))})
      );
    }

    out.println(
        "{}", format_row(columns, {"seed", fmt::format("{}{}", config.value("seed", 0), dtag(hints.default_seed))})
    );

    if (!hints.training_samples.empty()) {
      out.println("{}", format_row(columns, {"training samples", hints.training_samples}));
      out.println("{}", format_row(columns, {"test samples", hints.test_samples}));
    }

    if (config.contains("data")) {
      out.println("{}", format_row(columns, {"training data", config["data"].get<std::string>()}));
    }

    out.newline();
  }

  void print_data_summary(Output& out, nlohmann::json const& meta) {
    using namespace style;
    using namespace layout;

    std::vector<Column> columns = {
        {"Property", 18, Align::left},
        {"Value", 30, Align::left},
    };

    out.println("{}", emphasis("Data Summary"));
    out.newline();

    Row header = header_labels(columns);
    out.println("{}", format_row(columns, header));
    out.println("{}", muted(format_separator(columns)));

    if (meta.contains("observations")) {
      out.println("{}", format_row(columns, {"observations", std::to_string(meta["observations"].get<int>())}));
    }

    if (meta.contains("features")) {
      out.println("{}", format_row(columns, {"features", std::to_string(meta["features"].get<int>())}));
    }

    // Only emit the groups rows for classification models. Regression's
    // `meta.groups` is an empty array — printing "groups 0" would be noise.
    auto const group_names = meta.value("groups", types::Names{});
    if (!group_names.empty()) {
      out.println("{}", format_row(columns, {"groups", std::to_string(group_names.size())}));

      std::string names;

      for (std::size_t i = 0; i < group_names.size(); ++i) {
        if (i > 0) {
          names += ", ";
        }

        names += group_names[i];
      }

      out.println("{}", format_row(columns, {"group names", names}));
    }

    out.newline();
  }

  void print_summary(Output& out, nlohmann::json const& model_data, ConfigDisplayHints const& hints) {
    using namespace style;

    types::Names group_names;
    types::Names feature_names;

    if (model_data.contains("meta")) {
      auto const& meta = model_data["meta"];
      group_names      = meta.value("groups", types::Names{});
      feature_names    = meta.value("feature_names", types::Names{});
    }

    if (model_data.contains("config")) {
      print_configuration(out, model_data["config"], hints);
    }

    if (model_data.contains("meta")) {
      print_data_summary(out, model_data["meta"]);
    }

    if (model_data.contains("training_duration_ms")) {
      out.print("Trained in {}ms", emphasis(std::to_string(model_data["training_duration_ms"].get<long long>())));

      if (model_data.contains("save_path")) {
        out.print(", ");
        out.saved("model", model_data["save_path"].get<std::string>());
      } else {
        out.newline();
      }

      out.newline();
    }

    bool const is_degenerate = model_data.contains("model") && model_data["model"].value("degenerate", false);

    if (is_degenerate) {
      out.println("{} Some splits could not separate groups (degenerate nodes).", emphasis(warning("Warning:")));
      out.println("This can be caused by ill-conditioned variables in the input data,");
      out.println("or by bootstrap samples that produce singular covariance matrices.");
      out.println("Degenerate nodes predict the group with the most observations.");
      out.println("Degenerate trees are excluded from variable importance calculations.");
      out.newline();
    }


    types::Mode const mode = types::mode_from_string(model_data.at("config").at("mode").get<std::string>());

    print_metrics_from_json(out, model_data, "training_metrics", "Training", mode, group_names);
    print_metrics_from_json(out, model_data, "oob_metrics", "OOB", mode, group_names);

    // Variable importance
    if (has_value(model_data, "variable_importance")) {
      auto vi = model_data["variable_importance"].get<VariableImportance>();
      print_variable_importance(out, vi, feature_names);
    }
  }
}
