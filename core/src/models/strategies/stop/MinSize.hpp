#pragma once

#include "models/strategies/stop/StopRule.hpp"
#include "models/strategies/Strategy.hpp"

namespace ppforest2::stop {
  /**
   * @brief Stop when the node has fewer than a minimum number of observations.
   *
   */
  class MinSize : public StopRule {
  public:
    int min_size;

    explicit MinSize(int min_size);

    static StopRule::Ptr from_json(nlohmann::json const& j);

    nlohmann::json to_json() const override;
    std::string display_name() const override;
    std::set<types::Mode> supported_modes() const override {
      return {types::Mode::Classification, types::Mode::Regression};
    }

    PPFOREST2_REGISTER_STRATEGY(StopRule, "min_size")
    PPFOREST2_REGISTER_PRIMARY_PARAM("min_size", "min_size")

  protected:
    bool compute(NodeContext const& ctx, stats::RNG& rng) const override;
  };
}
