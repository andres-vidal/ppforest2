#pragma once

#include "models/strategies/binarize/Binarization.hpp"
#include "models/strategies/Strategy.hpp"
#include "utils/Types.hpp"

namespace ppforest2::binarize {
  /**
   * @brief Binarization by largest gap between sorted projected group means.
   *
   * Projects group means onto 1D via the projector, sorts by projected
   * mean, and finds the largest gap between consecutive means. Groups
   * below the gap become binary group 0, groups above become binary
   * group 1. This is the default PPtree binarization method.
   */
  class LargestGap : public Binarization {
  public:
    static Binarization::Ptr from_json(nlohmann::json const& j);

    nlohmann::json to_json() const override;
    std::string display_name() const override { return "Largest gap"; }
    std::set<types::Mode> supported_modes() const override { return {types::Mode::Classification}; }

    PPFOREST2_REGISTER_STRATEGY(Binarization, "largest_gap")

  protected:
    void compute(NodeContext& ctx, stats::RNG& rng) const override;
  };
}
