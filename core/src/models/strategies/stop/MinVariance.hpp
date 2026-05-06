#pragma once

#include "models/strategies/stop/StopRule.hpp"
#include "models/strategies/Strategy.hpp"

namespace ppforest2::stop {
  /**
   * @brief Stop when the response variance is below a threshold.
   *
   * Useful for regression trees. A node becomes a leaf when the
   * variance of the continuous response among its observations
   * falls below the specified threshold.
   *
   * Requires NodeContext::continuous_y to be set.
   */
  class MinVariance : public StopRule {
  public:
    types::Feature threshold;

    explicit MinVariance(types::Feature threshold);

    static StopRule::Ptr from_json(nlohmann::json const& j);

    nlohmann::json to_json() const override;
    std::string display_name() const override;
    std::set<types::Mode> supported_modes() const override { return {types::Mode::Regression}; }

    PPFOREST2_REGISTER_STRATEGY(StopRule, "min_variance")
    PPFOREST2_REGISTER_PRIMARY_PARAM("min_variance", "threshold")

  protected:
    bool compute(NodeContext const& ctx, stats::RNG& rng) const override;
  };
}
