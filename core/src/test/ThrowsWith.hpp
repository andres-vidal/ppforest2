#pragma once

#include <gtest/gtest.h>

#include <functional>
#include <stdexcept>
#include <string>

namespace ppforest2::test {

  namespace {
    inline std::string expected_message(std::string const& needle, std::string const& what) {
      return "expected message containing \"" + needle + "\", got \"" + what + "\"";
    }

    inline std::string wrong_exception(std::string const& what) {
      return "wrong exception type: " + what;
    }
  }

  /**
   * @brief gtest assertion: `fn` throws a `std::runtime_error` whose
   *        `what()` contains @p needle.
   *
   * Used by JSON-validation tests where the exception message carries the
   * dotted path of the offending field (e.g. `"config.pp.lambda: missing
   * required key"`). Centralised here so `JsonReader`, `ExportValidation`,
   * and per-strategy `from_json` tests share one helper.
   *
   * @code
   *   EXPECT_TRUE(throws_with([&] { reader.require<int>("k"); }, "k: missing"));
   * @endcode
   */
  inline ::testing::AssertionResult throws_with(std::function<void()> const& fn, std::string const& needle) {
    try {
      fn();
    } catch (std::runtime_error const& e) {
      std::string const what = e.what();
      if (what.find(needle) != std::string::npos) {
        return ::testing::AssertionSuccess();
      } else {
        return ::testing::AssertionFailure() << expected_message(needle, what);
      }
    } catch (std::exception const& e) {
      return ::testing::AssertionFailure() << wrong_exception(e.what());
    }
    return ::testing::AssertionFailure() << "did not throw";
  }
}
