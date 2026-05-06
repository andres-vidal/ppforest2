/**
 * @file Output.hpp
 * @brief Quiet-aware output context for CLI subcommands.
 *
 * Central output abstraction that handles indentation, quiet-mode
 * suppression, stderr error reporting, and progress display. All CLI
 * printing should go through an Output instance.
 */
#pragma once

#include "io/Color.hpp"

#include <fmt/format.h>
#include <cstdio>
#include <string>

namespace ppforest2::io {
  /**
   * @brief Quiet-aware, indentation-aware output context.
   *
   * Construct once per subcommand from Params::quiet and thread
   * through all printing functions. Stdout methods respect quiet mode
   * and prepend the current indentation. Stderr methods always print.
   */
  struct Output {
    bool quiet;
    int indent_level = 0;

    explicit Output(bool quiet)
        : quiet(quiet) {}

    // -- stdout (quiet-aware, indented) ----------------------------------

    /**
     * @brief Print indent + formatted content + newline. The workhorse.
     */
    template<typename... Args> void println(fmt::format_string<Args...> fmt_str, Args&&... args) const {
      if (!quiet) {
        print_indent();
        fmt::print(fmt_str, std::forward<Args>(args)...);
        fmt::print("\n");
      }
    }

    /**
     * @brief Print indent + formatted content, no newline.
     *
     * Use for starting a partial line that will be continued.
     */
    template<typename... Args> void print(fmt::format_string<Args...> fmt_str, Args&&... args) const {
      if (!quiet) {
        print_indent();
        fmt::print(fmt_str, std::forward<Args>(args)...);
      }
    }

    /**
     * @brief Print a blank line. For visual separation between sections.
     */
    void newline() const {
      if (!quiet) {
        fmt::print("\n");
      }
    }

    // -- indentation -----------------------------------------------------

    void indent() { ++indent_level; }

    void dedent() {
      if (indent_level > 0) {
        --indent_level;
      }
    }

    // -- high-level patterns ---------------------------------------------

    /**
     * @brief Print a file-save confirmation: "label saved to path".
     */
    void saved(std::string const& label, std::string const& path) const { println("{} saved to {}", label, path); }

    /**
     * @brief Display a carriage-return progress bar.
     *
     * Overwrites the current line with a bar showing current/total.
     * Prints a final newline when current == total.
     */
    void progress(int current, int total, int bar_width = 50) const {
      if (quiet) {
        return;
      }

      float pct = static_cast<float>(current) / static_cast<float>(total);
      int pos   = static_cast<int>(static_cast<float>(bar_width) * pct);

      std::string bar_tpl = style::emphasis("{} |");
      std::string bar     = std::string(pos, '-') + std::string(bar_width - pos, ' ');

      if (current == total) {
        bar_tpl = style::success(bar_tpl);
      } else {
        bar_tpl = style::info(bar_tpl);
      }

      fmt::print(
          "\r{:{}}" + bar_tpl + " {}/{} ({}%)     ",
          "",
          indent_level * 2,
          bar,
          current,
          total,
          static_cast<int>(pct * 100.0)
      );
      std::fflush(stdout);

      if (current == total) {
        fmt::print("\n");
      }
    }

  private:
    void print_indent() const {
      if (indent_level > 0) {
        fmt::print("{:{}}", "", indent_level * 2);
      }
    }
  };
}
