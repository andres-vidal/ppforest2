/**
 * @file BenchmarkReport.cpp
 * @brief Benchmark result table formatting, comparison display, and export.
 */
#include "cli/BenchmarkReport.hpp"
#include "io/Color.hpp"

#include <fmt/format.h>

namespace ppforest2::cli {
  namespace {
    std::string format_rss(std::optional<double> mb) {
      if (!mb) {
        return "N/A";
      }

      return fmt::format("{:.1f} MB", *mb);
    }

    std::string format_vars(ScenarioResult const& r) {
      if (r.size == 0) {
        return "--";
      }

      if (r.n_vars) {
        return fmt::format("{}", *r.n_vars);
      }

      if (r.p_vars) {
        return fmt::format("{:.2f}", *r.p_vars);
      }

      return "--";
    }

    std::string format_opt_delta(std::optional<double> const& val) {
      return val ? fmt::format("{:.1f}", *val) : "";
    }

    /**
     * @brief Format an error metric for display.
     *
     * Classification → percentage (`5.3%`). Regression MSE → raw with
     * three decimals (`12.345`).
     */
    std::string format_err(double v, types::Mode mode) {

      std::string res;

      if (mode == types::Mode::Regression) {
        res = fmt::format("{:.3f}", v);
      } else {
        res = fmt::format("{:.1f}%", v * 100);
      }

      return res;
    }
  }

  // --- BenchmarkReport ---

  std::vector<ppforest2::io::layout::Column> BenchmarkReport::make_columns(bool has_baseline) {
    using namespace ppforest2::io::layout;

    return {
        {"Scenario", 20, Align::left},
        {"n", 6, Align::right},
        {"p", 4, Align::right},
        {"g", 3, Align::right},
        {"size", 6, Align::right},
        {"vars", 5, Align::right},
        {"split", 5, Align::right},
        {"iters", 5, Align::right},
        {"Time (ms)", 18, Align::right},
        {"Δ Time", 8, Align::right, has_baseline},
        {"Peak RSS", 12, Align::right},
        {"Δ RSS", 8, Align::right, has_baseline},
        {"Train Err", 9, Align::right},
        {"Δ Train", 8, Align::right, has_baseline},
        {"Test Err", 8, Align::right},
        {"Δ Test", 8, Align::right, has_baseline},
    };
  }

  double BenchmarkReport::delta_pct(double current, double baseline) {
    return ((current - baseline) / baseline) * 100.0;
  }

  BenchmarkReport::Deltas BenchmarkReport::compute_deltas(ScenarioResult const& r, Baseline const& baseline) {
    Deltas d;
    auto match = baseline.find(r.name);

    if (!match) {
      return d;
    }

    auto const& b = match->get();

    if (b.mean_time_ms > 0) {
      d.time = delta_pct(r.mean_time_ms, b.mean_time_ms);
    }

    if (r.peak_rss_mb && b.peak_rss_mb && *b.peak_rss_mb > 0) {
      d.rss = delta_pct(*r.peak_rss_mb, *b.peak_rss_mb);
    }

    if (b.mean_tr_error > 0) {
      d.tr_err = delta_pct(r.mean_tr_error, b.mean_tr_error);
    }

    if (b.mean_te_error > 0) {
      d.te_err = delta_pct(r.mean_te_error, b.mean_te_error);
    }

    return d;
  }

  nlohmann::json BenchmarkReport::to_json() const {
    nlohmann::json j = result.to_json();

    if (baseline) {
      j["baseline_timestamp"] = baseline->timestamp;

      for (auto& rj : j["results"]) {
        std::string name = rj["name"];

        for (auto const& r : result.results) {
          if (r.name == name) {
            auto d = compute_deltas(r, *baseline);

            if (d.time) {
              rj["delta_time_pct"] = *d.time;
            }
            if (d.rss) {
              rj["delta_rss_pct"] = *d.rss;
            }
            if (d.tr_err) {
              rj["delta_train_error_pct"] = *d.tr_err;
            }
            if (d.te_err) {
              rj["delta_test_error_pct"] = *d.te_err;
            }

            break;
          }
        }
      }
    }

    return j;
  }

  std::string BenchmarkReport::to_csv() const {
    bool const has_baseline = baseline.has_value();

    std::string csv;

    csv += "scenario,n,p,g,trees,vars,train_ratio,runs,mean_time_ms,std_time_ms,"
           "mean_train_error,mean_test_error,peak_rss_mb,scenario_time_ms";

    if (has_baseline) {
      csv += ",delta_time_pct,delta_rss_pct,delta_train_error_pct,delta_test_error_pct";
    }

    csv += "\n";

    for (auto const& r : result.results) {
      csv += fmt::format(
          "{},{},{},{},{},{:.2f},{:.2f},{},{:.2f},{:.2f},{:.4f},{:.4f},{:.1f},{:.0f}",
          r.name,
          r.n,
          r.p,
          r.g,
          r.size,
          r.p_vars.value_or(0),
          r.train_ratio,
          r.runs,
          r.mean_time_ms,
          r.std_time_ms,
          r.mean_tr_error,
          r.mean_te_error,
          r.peak_rss_mb.value_or(-1.0),
          r.scenario_time_ms
      );

      if (has_baseline) {
        auto d = compute_deltas(r, *baseline);
        csv += fmt::format(
            ",{},{},{},{}",
            format_opt_delta(d.time),
            format_opt_delta(d.rss),
            format_opt_delta(d.tr_err),
            format_opt_delta(d.te_err)
        );
      }

      csv += "\n";
    }

    return csv;
  }

  void BenchmarkReport::print(io::Output& out, Style const& style) const {
    using namespace ppforest2::io::layout;

    bool const has_baseline = baseline.has_value();

    // Title
    out.newline();
    out.println("{}", style.format_title(result.suite_name, result.timestamp));

    if (has_baseline && !baseline->timestamp.empty()) {
      out.println("{}", style.format_baseline_label(baseline->timestamp));
    }

    out.newline();

    // Table
    auto columns = make_columns(has_baseline);
    out.println("{}", style.format_header(columns));

    auto decorate = [&](std::optional<double> const& val) -> std::string {
      if (!val) {
        return "";
      }

      std::string text = fmt::format("{:+.1f}%", *val);

      if (*val < -delta_threshold) {
        return style.good(text);
      } else if (*val > delta_threshold) {
        return style.bad(text);
      } else {
        return style.neutral(text);
      }
    };

    for (auto const& r : result.results) {
      Deltas d;

      if (has_baseline) {
        d = compute_deltas(r, *baseline);
      }

      Row cells = {
          r.name,
          fmt::format("{}", r.n),
          fmt::format("{}", r.p),
          fmt::format("{}", r.g),
          fmt::format("{}", r.size),
          r.size > 0 ? format_vars(r) : std::string("--"),
          fmt::format("{}%", static_cast<int>(r.train_ratio * 100)),
          fmt::format("{}", r.runs),
          fmt::format("{:.1f} ± {:.1f}", r.mean_time_ms, r.std_time_ms),
          decorate(d.time),
          format_rss(r.peak_rss_mb),
          decorate(d.rss),
          format_err(r.mean_tr_error, r.mode),
          decorate(d.tr_err),
          format_err(r.mean_te_error, r.mode),
          decorate(d.te_err),
      };

      out.println("{}", style.format_row(columns, cells));
    }

    // Footer
    out.newline();
    out.println("{}", style.format_footer(result.results.size(), result.total_time_ms));
  }

  // --- Text ---

  std::string BenchmarkReport::Text::good(std::string const& s) const {
    return ppforest2::io::style::success(s);
  }
  std::string BenchmarkReport::Text::bad(std::string const& s) const {
    return ppforest2::io::style::error(s);
  }
  std::string BenchmarkReport::Text::neutral(std::string const& s) const {
    return ppforest2::io::style::muted(s);
  }

  std::string BenchmarkReport::Text::format_title(std::string const& name, std::string const& ts) const {
    using namespace ppforest2::io::style;
    std::string title = emphasis(name);

    if (!ts.empty()) {
      title += " " + muted("(" + ts + ")");
    }

    return title;
  }

  std::string BenchmarkReport::Text::format_baseline_label(std::string const& ts) const {
    return fmt::format("Baseline: {}", ppforest2::io::style::muted(ts));
  }

  std::string BenchmarkReport::Text::format_header(Columns const& columns) const {
    using namespace ppforest2::io::style;
    using namespace ppforest2::io::layout;

    Row header = header_labels(columns);
    header[0]  = emphasis(header[0]);

    for (auto& h : header) {
      if (h.find("Δ") != std::string::npos) {
        h = muted(h);
      }
    }

    return ppforest2::io::layout::format_row(columns, header) + "\n" + muted(format_separator(columns));
  }

  std::string BenchmarkReport::Text::format_row(Columns const& columns, ppforest2::io::layout::Row const& cells) const {
    return ppforest2::io::layout::format_row(columns, cells);
  }

  std::string BenchmarkReport::Text::format_footer(std::size_t count, double total_ms) const {
    return fmt::format(
        "{} scenarios completed in {:.1f}s", ppforest2::io::style::emphasis(std::to_string(count)), total_ms / 1000.0
    );
  }

  // --- Markdown ---

  std::string BenchmarkReport::Markdown::good(std::string const& s) const {
    return fmt::format("🟢 {}", s);
  }
  std::string BenchmarkReport::Markdown::bad(std::string const& s) const {
    return fmt::format("🔴 {}", s);
  }
  std::string BenchmarkReport::Markdown::neutral(std::string const& s) const {
    return fmt::format("⚪ {}", s);
  }

  std::string BenchmarkReport::Markdown::format_title(std::string const& name, std::string const& ts) const {
    std::string title = fmt::format("## {}", name);

    if (!ts.empty()) {
      title += fmt::format("\n\nCurrent: {}", ts);
    }

    return title;
  }

  std::string BenchmarkReport::Markdown::format_baseline_label(std::string const& ts) const {
    return fmt::format("Baseline: {}", ts);
  }

  std::string BenchmarkReport::Markdown::format_header(Columns const& columns) const {
    using namespace ppforest2::io::layout;
    return format_md_row(header_labels(columns), columns) + "\n" + format_md_separator(columns);
  }

  std::string
  BenchmarkReport::Markdown::format_row(Columns const& columns, ppforest2::io::layout::Row const& cells) const {
    return ppforest2::io::layout::format_md_row(cells, columns);
  }

  std::string BenchmarkReport::Markdown::format_footer(std::size_t count, double total_ms) const {
    return fmt::format("{} scenarios completed in {:.1f}s", count, total_ms / 1000.0);
  }
}
