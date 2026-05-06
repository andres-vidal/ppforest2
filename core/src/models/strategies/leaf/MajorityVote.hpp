#pragma once

#include "models/strategies/leaf/LeafStrategy.hpp"
#include "models/strategies/Strategy.hpp"

namespace ppforest2::leaf {
  /**
   * @brief Leaf creation by majority vote.
   *
   * Returns a TreeLeaf whose label is the most frequent class
   * in the node's group partition. When there is only one group,
   * it is returned directly. On ties, the numerically smallest
   * group label wins (deterministic).
   */
  class MajorityVote : public LeafStrategy {
  public:
    static LeafStrategy::Ptr from_json(nlohmann::json const& j);

    nlohmann::json to_json() const override;
    std::string display_name() const override { return "Majority vote"; }
    std::set<types::Mode> supported_modes() const override { return {types::Mode::Classification}; }

    PPFOREST2_REGISTER_STRATEGY(LeafStrategy, "majority_vote")

  protected:
    /**
     * @brief Create a majority-vote leaf from the node's group partition.
     */
    TreeNode::Ptr compute(NodeContext const& ctx, stats::RNG& rng) const override;
  };
}
