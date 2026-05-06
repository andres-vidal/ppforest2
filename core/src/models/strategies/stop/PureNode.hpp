#pragma once

#include "models/strategies/stop/StopRule.hpp"
#include "models/strategies/Strategy.hpp"

namespace ppforest2::stop {
  /**
   * @brief Stop when the node contains only one group (pure node).
   *
   * This is the default PPtree stopping rule: a node becomes a leaf
   * when all its observations belong to the same class.
   */
  class PureNode : public StopRule {
  public:
    static StopRule::Ptr from_json(nlohmann::json const& j);

    nlohmann::json to_json() const override;
    std::string display_name() const override { return "Pure node"; }
    std::set<types::Mode> supported_modes() const override { return {types::Mode::Classification}; }

    PPFOREST2_REGISTER_STRATEGY(StopRule, "pure_node")

  protected:
    bool compute(NodeContext const& ctx, stats::RNG& rng) const override;
  };
}
