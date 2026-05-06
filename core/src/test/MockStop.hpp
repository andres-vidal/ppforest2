#pragma once

#include "models/strategies/stop/StopRule.hpp"

#include <nlohmann/json.hpp>

#include <set>
#include <string>

namespace ppforest2::test {
  /**
   * @brief Mock `StopRule` whose verdict is fixed at construction time.
   *
   * Decouples tests that exercise composition or NVI plumbing from any
   * real stop rule. Carries a `compute_calls` counter so callers can
   * assert how many times the inner `compute` was invoked — useful for
   * verifying short-circuit semantics in `CompositeStop`.
   */
  class MockStop : public stop::StopRule {
  public:
    bool compute_return;
    mutable int compute_calls = 0;

    explicit MockStop(bool compute_return = false)
        : compute_return(compute_return) {}

    nlohmann::json to_json() const override { return {{"name", "mock_stop"}}; }
    std::string display_name() const override { return "Mock Stop"; }
    std::set<types::Mode> supported_modes() const override {
      return {types::Mode::Classification, types::Mode::Regression};
    }

  protected:
    bool compute(NodeContext const& /*ctx*/, stats::RNG& /*rng*/) const override {
      ++compute_calls;
      return compute_return;
    }
  };
}
