/**
 * @file Presentation.cpp
 * @brief Model statistics, variable importance serialization, and
 *        formatted terminal display for confusion matrices and VI tables.
 */
#include "io/Presentation.hpp"
#include "io/Table.hpp"
#include "serialization/Json.hpp"

#include <fmt/format.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace ppforest2::io {
  nlohmann::json ModelStats::to_json() const {
    nlohmann::json j = {
      { "runs",             tr_times.size() },
      { "mean_time_ms",     mean_time() },
      { "std_time_ms",      std_time() },
      { "mean_train_error", mean_tr_error() },
      { "std_train_error",  std_tr_error() },
      { "mean_test_error",  mean_te_error() },
      { "std_test_error",   std_te_error() }
    };

    if (peak_rss_bytes >= 0) {
      j["peak_rss_bytes"] = peak_rss_bytes;
      j["peak_rss_mb"]    = static_cast<double>(peak_rss_bytes) / (1024.0 * 1024.0);
    }

    // Per-iteration data
    nlohmann::json iterations = nlohmann::json::array();
    for (int i = 0; i < tr_times.size(); ++i) {
      iterations.push_back({
        { "train_time_ms", tr_times[i] },
        { "train_error",   tr_error[i] },
        { "test_error",    te_error[i] }
      });
    }

    j["iterations"] = iterations;

    return j;
  }

  void print_results(Output& out, const ModelStats& stats) {
    using namespace style;
    using namespace layout;

    out.println("{}", emphasis("Evaluation results"));
    out.newline();

    std::vector<Column> columns = {
      { "Runs",       5,  Align::left },
      { "Time (ms)", 18,  Align::right },
      { "Train Err", 10,  Align::right },
      { "Test Err",  10,  Align::right },
    };

    if (stats.peak_rss_bytes >= 0) {
      columns.push_back({ "Peak RSS", 10, Align::right });
    }

    Row header = header_labels(columns);

    out.println("{}", format_row(columns, header));
    out.println("{}", muted(format_separator(columns)));

    std::string time_str = fmt::format("{:.2f} +/- {:.2f}", stats.mean_time(), stats.std_time());
    std::string tr_err   = fmt::format("{:.2f}%", stats.mean_tr_error() * 100);
    std::string te_err   = fmt::format("{:.2f}%", stats.mean_te_error() * 100);

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
    Output&                         out,
    const VariableImportance&       vi,
    const std::vector<std::string>& feature_names,
    int                             max_rows) {
    using namespace style;
    using namespace layout;

    const auto& vi1   = vi.permuted;
    const auto& vi2   = vi.projections;
    const auto& vi3   = vi.weighted_projections;
    const auto& scale = vi.scale;

    const int p = static_cast<int>(vi2.size());

    bool has_names = static_cast<int>(feature_names.size()) == p;

    std::vector<int> order(static_cast<std::size_t>(p));
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&vi2](int a, int b) {
        return std::isgreater(vi2(a), vi2(b));
      });

    const bool show_vi1 = vi1.size() == vi2.size();
    const bool show_vi3 = vi3.size() == vi2.size();
    const int rows      = (max_rows > 0 && p > max_rows) ? max_rows : p;

    // Compute variable column width based on longest name.
    int var_width = 10;

    if (has_names) {
      for (int rank = 0; rank < rows; ++rank) {
        int j   = order[static_cast<std::size_t>(rank)];
        int len = static_cast<int>(feature_names[static_cast<std::size_t>(j)].size());

        if (len + 1 > var_width) var_width = len + 1;
      }
    }

    out.println("{}", emphasis("Variable Importance:"));
    out.newline();

    // Build columns conditionally
    std::vector<Column> columns = {
      { "Variable",   var_width, Align::left },
      { "",           10,        Align::right },
      { "Projection", 12,        Align::right },
    };

    if (show_vi3) columns.push_back({ "Weighted", 12, Align::right });

    if (show_vi1) columns.push_back({ "Permuted", 12, Align::right });

    // Header
    Row header = header_labels(columns);
    header[0] = emphasis(header[0]);
    header[1] = muted(fmt::format("{}", "\xcf\x83"));
    header[2] = emphasis(header[2]);

    if (show_vi3) header[3] = emphasis(header[3]);

    if (show_vi1) header[show_vi3 ? 4 : 3] = emphasis(header[show_vi3 ? 4 : 3]);

    out.println("{}", format_row(columns, header));
    out.println("{}", muted(format_separator(columns)));

    // Data rows
    for (int rank = 0; rank < rows; ++rank) {
      int j         = order[static_cast<std::size_t>(rank)];
      std::string v = has_names
        ? feature_names[static_cast<std::size_t>(j)]
        : fmt::format("x{}", j + 1);

      Row cells = {
        v,
        fmt::format("{:.4f}", scale(j)),
        fmt::format("{:.6f}", vi2(j)),
      };

      if (show_vi3) cells.push_back(fmt::format("{:.6f}", vi3(j)));

      if (show_vi1) cells.push_back(fmt::format("{:.6f}", vi1(j)));

      out.println("{}", format_row(columns, cells));
    }

    if (rows < p) {
      out.println("{}", muted(fmt::format("... {} more variables not shown", p - rows)));
    }

    bool all_ones = (scale.array() - types::Feature(1)).abs().maxCoeff() < types::Feature(1e-6);

    if (!all_ones) {
      out.newline();
      out.newline();
      out.println("{} Variable importance was calculated using scaled coefficients (|a_j| * \u03c3_j).", emphasis(warning("Note:")));
      out.println("Variable contributions can only be theoretically interpreted as such");
      out.println("if the model was trained on scaled data. Scaling also changes the");
      out.println("projection-pursuit optimization, which may affect the resulting tree.");
    }

    out.newline();
  }

  void print_confusion_matrix(
    Output&                         out,
    const stats::ConfusionMatrix&   cm,
    const std::string&              title,
    const std::vector<std::string>& group_names) {
    using namespace style;

    auto group_err = cm.group_errors();

    bool has_names = !group_names.empty();

    // Compute column width: max of group name lengths and default width (5).
    int col_width = 5;

    if (has_names) {
      for (const auto& [label, idx] : cm.label_index) {
        int name_len = static_cast<int>(group_names[static_cast<std::size_t>(label)].size());

        if (name_len + 1 > col_width) {
          col_width = name_len + 1;
        }
      }
    }

    // Compute row label width.
    int row_label_width = 4;

    if (has_names) {
      for (const auto& [label, idx] : cm.label_index) {
        int name_len = static_cast<int>(group_names[static_cast<std::size_t>(label)].size());

        if (name_len > row_label_width) {
          row_label_width = name_len;
        }
      }
    }

    out.println("{}", emphasis(title + ":"));
    out.newline();

    // Header row: group labels
    std::string header(static_cast<std::size_t>(row_label_width), ' ');

    for (const auto& [label, idx] : cm.label_index) {
      std::string name = has_names ? group_names[static_cast<std::size_t>(label)] : std::to_string(label);
      header += fmt::format("{:>{}}", name, col_width);
    }

    header += "  Error";
    out.println("{}", header);

    // Data rows
    for (const auto& [label, row_idx] : cm.label_index) {
      std::string row_label = has_names ? group_names[static_cast<std::size_t>(label)] : std::to_string(label);
      std::string row       = fmt::format("{:>{}}", row_label, row_label_width);

      for (const auto& [col_label, col_idx] : cm.label_index) {
        int val          = cm.values(row_idx, col_idx);
        std::string cell = fmt::format("{:>{}}", val, col_width - 1);

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

  void print_configuration(
    Output&                   out,
    const nlohmann::json&     config,
    const ConfigDisplayHints& hints) {
    using namespace style;
    using namespace layout;

    auto dtag = [&](bool is_default) -> std::string {
        return is_default ? " " + muted("(default)") : "";
      };

    std::string model_type = config.value("trees", 0) > 0
      ? "Random Forest of Projection-Pursuit Oblique Decision Trees"
      : "Projection-Pursuit Oblique Decision Tree";

    out.println("{}", emphasis(model_type));
    out.newline();

    std::vector<Column> columns = {
      { "Parameter", 18, Align::left },
      { "Value",     30, Align::left },
    };

    Row header = header_labels(columns);
    out.println("{}", format_row(columns, header));
    out.println("{}", muted(format_separator(columns)));

    int trees = config.value("trees", 0);

    if (trees > 0) {
      out.println("{}", format_row(columns, { "trees", std::to_string(trees) }));

      if (config.contains("vars")) {
        std::string vars_str = std::to_string(config["vars"].get<int>());

        if (hints.vars_percent >= 0) {
          vars_str += fmt::format(" ({}%)", hints.vars_percent);
        }

        vars_str += dtag(hints.default_vars);
        out.println("{}", format_row(columns, { "variables/split", vars_str }));
      }

      out.println("{}", format_row(columns, { "threads", fmt::format("{}{}", config.value("threads", 1), dtag(hints.default_threads)) }));
      out.println("{}", format_row(columns, { "seed", fmt::format("{}{}", config.value("seed", 0), dtag(hints.default_seed)) }));
    }

    float lambda       = config.value("lambda", 0.5f);
    std::string method = lambda == 0 ? "LDA" : "PDA";
    out.println("{}", format_row(columns, { "method", fmt::format("{} (lambda={})", method, lambda) }));

    if (!hints.training_samples.empty()) {
      out.println("{}", format_row(columns, { "training samples", hints.training_samples }));
      out.println("{}", format_row(columns, { "test samples", hints.test_samples }));
    }

    if (config.contains("data")) {
      out.println("{}", format_row(columns, { "training data", config["data"].get<std::string>() }));
    }

    out.newline();
  }

  void print_data_summary(
    Output &              out,
    const nlohmann::json& meta) {
    using namespace style;
    using namespace layout;

    std::vector<Column> columns = {
      { "Property", 18, Align::left },
      { "Value",    30, Align::left },
    };

    out.println("{}", emphasis("Data Summary"));
    out.newline();

    Row header = header_labels(columns);
    out.println("{}", format_row(columns, header));
    out.println("{}", muted(format_separator(columns)));

    if (meta.contains("observations")) {
      out.println("{}", format_row(columns, { "observations", std::to_string(meta["observations"].get<int>()) }));
    }

    if (meta.contains("features")) {
      out.println("{}", format_row(columns, { "features", std::to_string(meta["features"].get<int>()) }));
    }

    if (meta.contains("groups")) {
      auto group_names = meta["groups"].get<std::vector<std::string>>();
      out.println("{}", format_row(columns, { "groups", std::to_string(group_names.size()) }));

      std::string names;

      for (std::size_t i = 0; i < group_names.size(); ++i) {
        if (i > 0) names += ", ";

        names += group_names[i];
      }

      out.println("{}", format_row(columns, { "group names", names }));
    }

    out.newline();
  }

  void print_summary(
    Output&                   out,
    const nlohmann::json&     model_data,
    const ConfigDisplayHints& hints) {
    using namespace style;

    std::vector<std::string> group_names;
    std::vector<std::string> feature_names;

    if (model_data.contains("meta")) {
      const auto& meta = model_data["meta"];

      if (meta.contains("groups")) {
        group_names = meta["groups"].get<std::vector<std::string>>();
      }

      if (meta.contains("feature_names")) {
        feature_names = meta["feature_names"].get<std::vector<std::string>>();
      }
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

    bool is_degenerate = model_data.contains("model")
      && model_data["model"].value("degenerate", false);

    if (is_degenerate) {
      out.println("{} Some splits could not separate groups (degenerate nodes).", emphasis(warning("Warning:")));
      out.println("This can be caused by ill-conditioned variables in the input data,");
      out.println("or by bootstrap samples that produce singular covariance matrices.");
      out.println("Degenerate nodes predict the group with the most observations.");
      out.println("Degenerate trees are excluded from variable importance calculations.");
      out.newline();
    }

    // Training confusion matrix
    if (model_data.contains("training_confusion_matrix")) {
      auto cm = serialization::confusion_matrix_from_json(model_data["training_confusion_matrix"]);
      out.println("{} {}", emphasis("Training Error:"), fmt::format("{:.2f}%", cm.error() * 100));
      print_confusion_matrix(out, cm, "Training Confusion Matrix", group_names);
    }

    // OOB error and confusion matrix
    if (model_data.contains("oob_error")) {
      out.println("{} {}", emphasis("OOB Error:"), fmt::format("{:.2f}%", model_data["oob_error"].get<double>() * 100));
    }

    if (model_data.contains("oob_confusion_matrix")) {
      auto cm = serialization::confusion_matrix_from_json(model_data["oob_confusion_matrix"]);
      print_confusion_matrix(out, cm, "OOB Confusion Matrix", group_names);
    } else {
      out.newline();
    }

    // Variable importance
    if (model_data.contains("variable_importance")) {
      auto vi = serialization::variable_importance_from_json(model_data["variable_importance"]);
      print_variable_importance(out, vi, feature_names);
    }
  }
}
