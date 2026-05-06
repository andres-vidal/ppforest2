#pragma once

#include "models/strategies/stop/StopRule.hpp"
#include "models/strategies/Strategy.hpp"

namespace ppforest2::stop {
  /**
   * @brief Stop when the node's depth reaches a configured maximum.
   *
   * Depth is zero-based at the root. `max_depth(k)` allows the tree to
   * have at most `k + 1` levels (root at depth 0, leaves at depth k).
   * Useful for bounding tree complexity independently of sample size
   * and response variance — the two other regression-oriented stop
   * rules in this family. Mode-agnostic.
   */
  class MaxDepth : public StopRule {
  public:
    int max_depth;

    explicit MaxDepth(int max_depth);

    static StopRule::Ptr from_json(nlohmann::json const& j);

    nlohmann::json to_json() const override;
    std::string display_name() const override;
    std::set<types::Mode> supported_modes() const override {
      return {types::Mode::Classification, types::Mode::Regression};
    }

    PPFOREST2_REGISTER_STRATEGY(StopRule, "max_depth")
    PPFOREST2_REGISTER_PRIMARY_PARAM("max_depth", "max_depth")

  protected:
    bool compute(NodeContext const& ctx, stats::RNG& rng) const override;
  };
}
