/**
 * @file Color.hpp
 * @brief TTY-aware colored terminal output utilities.
 *
 * Wraps the fmt/color.h library with conditional formatting that
 * automatically disables ANSI escape codes when stdout is not a TTY
 * or when the user passes the --no-color flag. On Windows, enables
 * Virtual Terminal Processing for ANSI support.
 */
#pragma once

#include <string>

#include <fmt/color.h>
#include <fmt/format.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

/**
 * @brief TTY-aware ANSI colored output utilities.
 *
 * Wraps fmt/color.h with conditional formatting that disables escape
 * codes when stdout is not a TTY or the user passes --no-color.
 * Provides semantic helpers: error(), success(), emphasis(), muted(),
 * info(), and warning().
 */
namespace ppforest2::io::style {
  /**
   * @brief Global toggle for colored output.
   * @return Reference to the static enabled flag.
   */
  inline bool& color_enabled() {
    static bool enabled = true;
    return enabled;
  }

  /**
   * @brief Initialize color support based on TTY detection and user preference.
   *
   * On Windows, enables ANSI Virtual Terminal Processing. Disables color
   * when stdout is not a TTY or when the user explicitly requests no color.
   *
   * @param no_color If true, unconditionally disable colored output.
   */
  inline void init_color(bool no_color) {
    if (no_color) {
      color_enabled() = false;
      return;
    }
    // clang-format off
    #ifdef _WIN32
    // clang-format on
    HANDLE hOut  = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD dwMode = 0;

    if (hOut != INVALID_HANDLE_VALUE && GetConsoleMode(hOut, &dwMode)) {
      SetConsoleMode(hOut, dwMode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
      color_enabled() = true;
    } else {
      color_enabled() = false;
    }
    // clang-format off
    #else
    color_enabled() = static_cast<bool>(isatty(fileno(stdout)));
    #endif
    // clang-format on
  }

  /**
   * @brief Apply a foreground color with a specific reset (\033[39m)
   * instead of fmt's full reset, so that emphasis and other
   * non-fg styles survive nesting.
   */
  inline std::string styled(std::string const& s, fmt::terminal_color c) {
    if (!color_enabled()) {
      return s;
    }

    return fmt::format("\033[{}m{}\033[39m", static_cast<uint8_t>(c), s);
  }

  /**
   * @brief Format text in red (for error messages).
   */
  inline std::string error(std::string const& s) {
    return styled(s, fmt::terminal_color::red);
  }

  /**
   * @brief Format text in green (for success messages).
   */
  inline std::string success(std::string const& s) {
    return styled(s, fmt::terminal_color::green);
  }

  /**
   * @brief Format text in bold (for emphasis / labels).
   *
   * Uses a specific bold-off code (\033[22m) so that fg colors survive
   * nesting, e.g. success("total: " + emphasis("7")).
   */
  inline std::string emphasis(std::string const& s) {
    if (!color_enabled()) {
      return s;
    }

    return fmt::format("\033[1m{}\033[22m", s);
  }

  /**
   * @brief Format text in dim gray (for hints and secondary info).
   */
  inline std::string muted(std::string const& s) {
    return styled(s, fmt::terminal_color::bright_black);
  }

  /**
   * @brief Format text in cyan (for informational highlights like progress bars).
   */
  inline std::string info(std::string const& s) {
    return styled(s, fmt::terminal_color::cyan);
  }

  /**
   * @brief Format text in yellow (for warnings).
   */
  inline std::string warning(std::string const& s) {
    return styled(s, fmt::terminal_color::yellow);
  }

  /**
   * @brief Wrap @p text in an OSC 8 hyperlink targeting @p url.
   *
   * Supported by iTerm2, WezTerm, Konsole, modern xterm, Windows Terminal
   * and others. In terminals that don't recognize the escape sequence the
   * URL text still shows verbatim, so the only behavioral difference is
   * "click to open" in capable terminals. Falls back to plain @p text
   * when colored output is disabled (no-TTY or --no-color).
   */
  inline std::string link(std::string const& url, std::string const& text) {
    if (!color_enabled()) {
      return text;
    }
    // BEL terminator (rather than ST `ESC \`) — more widely accepted by
    // terminals that implement only a subset of the OSC 8 spec.
    return fmt::format("\x1b]8;;{}\x07{}\x1b]8;;\x07", url, text);
  }
}
