/**
 * @file BenchmarkReport.hpp
 * @brief Benchmark result formatting, comparison, and export.
 */
#pragma once

#include "cli/Benchmark.hpp"
#include "io/Output.hpp"
#include "io/Table.hpp"
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace ppforest2::cli {
  /**
   * @brief A baseline suite with a pre-built name → result index.
   *
   * Constructed from a SuiteResult; builds the lookup index once
   * so print functions can match scenarios by name efficiently.
   */
  struct Baseline : SuiteResult {
    explicit Baseline(SuiteResult suite)
        : SuiteResult(std::move(suite)) {
      for (auto const& r : results) {
        index_[r.name] = &r;
      }
    }

    /** @brief Find a scenario by name. */
    std::optional<std::reference_wrapper<ScenarioResult const>> find(std::string const& name) const {
      auto it = index_.find(name);

      if (it == index_.end()) {
        return std::nullopt;
      }

      return std::cref(*it->second);
    }

  private:
    std::unordered_map<std::string, ScenarioResult const*> index_;
  };

  /**
   * @brief Benchmark report with comparison, display, and export.
   *
   * Constructed with the suite result and an optional baseline.
   * Display style (terminal or markdown) is passed to `print`.
   * Export methods (`to_json`, `to_csv`) are format-independent.
   */
  struct BenchmarkReport {
    double delta_threshold = 1.0;

    explicit BenchmarkReport(SuiteResult const& r, std::optional<Baseline> const& b = std::nullopt)
        : result(r)
        , baseline(b) {}

    /**
     * @brief Display style for benchmark report output.
     *
     * Subclasses provide decoration (colors/emoji) and layout
     * (aligned columns vs markdown tables).
     */
    struct Style {
      virtual ~Style() = default;

    protected:
      friend struct BenchmarkReport;

      virtual std::string good(std::string const& s) const    = 0;
      virtual std::string bad(std::string const& s) const     = 0;
      virtual std::string neutral(std::string const& s) const = 0;

      using Columns = std::vector<ppforest2::io::layout::Column>;

      virtual std::string format_title(std::string const& name, std::string const& timestamp) const         = 0;
      virtual std::string format_baseline_label(std::string const& timestamp) const                         = 0;
      virtual std::string format_header(Columns const& columns) const                                       = 0;
      virtual std::string format_row(Columns const& columns, ppforest2::io::layout::Row const& cells) const = 0;
      virtual std::string format_footer(std::size_t count, double total_ms) const                           = 0;
    };

    /** @brief Terminal style with ANSI colors. */
    struct Text;

    /** @brief GitHub-flavored markdown style with emoji indicators. */
    struct Markdown;

    /** @brief Print formatted results to the given output. */
    void print(io::Output& out, Style const& style) const;

    /** @brief Export report as JSON, including deltas when baseline is present. */
    nlohmann::json to_json() const;

    /** @brief Export report as CSV, including delta columns when baseline is present. */
    std::string to_csv() const;

  private:
    SuiteResult const& result;
    std::optional<Baseline> const& baseline;

    static double delta_pct(double current, double baseline);
    static std::vector<ppforest2::io::layout::Column> make_columns(bool has_baseline);

    struct Deltas {
      std::optional<double> time;
      std::optional<double> rss;
      std::optional<double> tr_err;
      std::optional<double> te_err;
    };

    static Deltas compute_deltas(ScenarioResult const& r, Baseline const& baseline);
  };

  /** @brief Terminal style with ANSI color decorators. */
  struct BenchmarkReport::Text : BenchmarkReport::Style {
    std::string good(std::string const& s) const override;
    std::string bad(std::string const& s) const override;
    std::string neutral(std::string const& s) const override;

    std::string format_title(std::string const& name, std::string const& ts) const override;
    std::string format_baseline_label(std::string const& ts) const override;
    std::string format_header(Columns const& columns) const override;
    std::string format_row(Columns const& columns, ppforest2::io::layout::Row const& cells) const override;
    std::string format_footer(std::size_t count, double total_ms) const override;
  };

  /** @brief Markdown style with emoji indicators. */
  struct BenchmarkReport::Markdown : BenchmarkReport::Style {
    std::string good(std::string const& s) const override;
    std::string bad(std::string const& s) const override;
    std::string neutral(std::string const& s) const override;

    std::string format_title(std::string const& name, std::string const& ts) const override;
    std::string format_baseline_label(std::string const& ts) const override;
    std::string format_header(Columns const& columns) const override;
    std::string format_row(Columns const& columns, ppforest2::io::layout::Row const& cells) const override;
    std::string format_footer(std::size_t count, double total_ms) const override;
  };
}
