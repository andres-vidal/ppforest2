#pragma once

#include "models/strategies/grouping/Grouping.hpp"
#include "models/strategies/Strategy.hpp"

namespace ppforest2::grouping {
  /**
   * @brief Label-based grouping: create partitions from sorted class labels.
   *
   * This is the default classification grouping strategy.
   *
   * - `init()`: wraps sorted integer labels into a GroupPartition.
   * - `split()`: routes all observations of a group to the same child
   *   based on the binary regrouping. Uses the subgroups mapping to
   *   recover original class labels for each child.
   */
  class ByLabel : public Grouping {
  public:
    static Grouping::Ptr from_json(nlohmann::json const& j);

    nlohmann::json to_json() const override;
    std::string display_name() const override { return "By label"; }
    std::set<types::Mode> supported_modes() const override { return {types::Mode::Classification}; }

    PPFOREST2_REGISTER_STRATEGY(Grouping, "by_label")

  protected:
    /**
     * @brief Create a GroupPartition from sorted integer-valued labels.
     *
     * Casts the float-typed `y` to `GroupIdVector` internally.
     */
    stats::GroupPartition compute_init(types::OutcomeVector const& y) const override;

    /**
     * @brief Subclass implementation of per-node splitting.
     *
     * Writes `ctx.lower_y_part` and `ctx.upper_y_part`.
     */
    void compute(NodeContext& ctx, stats::RNG& rng) const override;
  };
}
