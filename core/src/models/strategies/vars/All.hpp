#pragma once

#include "models/strategies/vars/VariableSelection.hpp"
#include "models/strategies/Strategy.hpp"

namespace ppforest2::vars {
  /**
   * @brief Selects all variables (no variable selection).
   *
   * Used with standard (non-random-forest) trees where all features
   * are available to the projection pursuit step at every node.
   */
  class All : public VariableSelection {
  protected:
    void compute(NodeContext& ctx, stats::RNG& rng) const override;

  public:
    nlohmann::json to_json() const override;
    std::string display_name() const override { return "All variables"; }
    std::set<types::Mode> supported_modes() const override {
      return {types::Mode::Classification, types::Mode::Regression};
    }

    static VariableSelection::Ptr from_json(nlohmann::json const& j);

    PPFOREST2_REGISTER_STRATEGY(VariableSelection, "all")
  };
}
