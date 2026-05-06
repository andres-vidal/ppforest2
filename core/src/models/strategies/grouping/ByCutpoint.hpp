#pragma once

#include "models/strategies/grouping/Grouping.hpp"
#include "models/strategies/Strategy.hpp"

namespace ppforest2::grouping {
  /**
   * @brief Cutpoint-based grouping for regression trees.
   *
   * At each node, observations are split by the cutpoint in projected space,
   * then each child's observations are sorted by continuous_y and
   * median-split into 2 artificial groups for the next PDA projection.
   *
   * Requires the non-const ctx.x and ctx.y_vec on NodeContext for in-place
   * reordering of data within each node's range.
   *
   * - `init()`: wraps pre-computed GroupIdVector (median-split of sorted y)
   *   into a GroupPartition (same as ByLabel).
   * - `split()`: partitions observations by cutpoint, re-sorts each child
   *   by continuous_y, median-splits into 2 groups.
   */
  class ByCutpoint : public Grouping {
  public:
    static Grouping::Ptr from_json(nlohmann::json const& j);

    nlohmann::json to_json() const override;
    std::string display_name() const override { return "By cutpoint"; }
    std::set<types::Mode> supported_modes() const override { return {types::Mode::Regression}; }

    PPFOREST2_REGISTER_STRATEGY(Grouping, "by_cutpoint")

  protected:
    /**
     * @brief Median-split the sorted continuous response into 2 groups.
     *
     * The caller must pre-sort `y` ascending. For `n >= 2` the result is
     * `GroupPartition(0, n - 1).bisect(n / 2)`; for `n == 1` a single-group
     * partition covering `[0, 0]` is returned.
     */
    stats::GroupPartition compute_init(types::OutcomeVector const& y) const override;

    /**
     * @brief Split observations by cutpoint and re-cluster each child.
     *
     * Requires the non-const ctx.x and ctx.y_vec on the context.
     * 1. Partitions rows within the node's range by projected value vs cutpoint.
     * 2. Sorts each child's rows by continuous_y.
     * 3. Median-splits each child into 2 groups.
     */
    void compute(NodeContext& ctx, stats::RNG& rng) const override;
  };
}
