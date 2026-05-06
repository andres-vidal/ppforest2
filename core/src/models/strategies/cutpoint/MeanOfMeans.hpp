#pragma once

#include "models/strategies/cutpoint/Cutpoint.hpp"
#include "models/strategies/Strategy.hpp"
#include "utils/Types.hpp"

namespace ppforest2::cutpoint {
  /**
   * @brief Split cutpoint as the mean of two group means.
   *
   * Computes the midpoint between the projected means of the two
   * groups: (mean(group_1 * A) + mean(group_2 * A)) / 2.
   * This is the default rule used by PPforest.
   */
  class MeanOfMeans : public Cutpoint {
  public:
    static Cutpoint::Ptr from_json(nlohmann::json const& j);

    nlohmann::json to_json() const override;
    std::string display_name() const override { return "Mean of means"; }
    std::set<types::Mode> supported_modes() const override {
      return {types::Mode::Classification, types::Mode::Regression};
    }

    PPFOREST2_REGISTER_STRATEGY(Cutpoint, "mean_of_means")

  protected:
    void compute(NodeContext& ctx, stats::RNG& rng) const override;
  };
}
