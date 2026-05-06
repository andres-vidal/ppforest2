#pragma once

#include "models/strategies/Strategy.hpp"
#include "stats/GroupPartition.hpp"
#include "stats/Stats.hpp"
#include "utils/Types.hpp"

/**
 * @brief Grouping strategies that manage group partitions throughout training.
 *
 * The Grouping strategy owns the full lifecycle of GroupPartitions:
 * initial construction from training labels (`init`) and per-node
 * child splitting (`split`).
 *
 * For classification, ByLabel constructs from sorted labels and routes
 * groups to children via the binary mapping.  For regression (future),
 * ByCutpoint quantile-slices the continuous response and re-clusters
 * children at each node.
 */
namespace ppforest2 {
  struct NodeContext;
}

namespace ppforest2::grouping {
  /**
   * @brief Abstract strategy for managing group partitions.
   *
   * `init()` builds the root partition once before training; `split()`
   * writes `ctx.lower_y_part` and `ctx.upper_y_part` per node.
   */
  class Grouping : public Strategy<Grouping> {
  public:
    /**
     * @brief Create the initial GroupPartition from the training response.
     *
     * Caller must pre-sort rows so equal values (classification) or ascending
     * response (regression) form contiguous blocks.
     *
     * Public NVI entry point: delegates to the subclass-provided
     * `compute_init`, then asserts the partition's post-conditions —
     * blocks rooted at row 0 and covering all of `y`. Downstream callers
     * (e.g. stratified resampling in `ClassificationForest::train_tree`)
     * rely on those properties to safely reuse the parent partition over
     * a resampled response.
     */
    stats::GroupPartition init(types::OutcomeVector const& y) const;

    /** @brief Convenience overload for callers with a `GroupIdVector`. */
    stats::GroupPartition init(types::GroupIdVector const& y) const {
      return init(types::OutcomeVector(y.cast<types::Outcome>()));
    }

    /**
     * @brief Split observations into two child partitions; writes
     *        ctx.lower_y_part / upper_y_part.
     *
     * Public NVI entry point: skips if `ctx.aborted` is set, otherwise
     * delegates to the subclass-provided `compute`.
     */
    void split(NodeContext& ctx, stats::RNG& rng) const;

  protected:
    /** @brief Subclass implementation of initial partition construction. */
    virtual stats::GroupPartition compute_init(types::OutcomeVector const& y) const = 0;

    /** @brief Subclass implementation of per-node splitting. */
    virtual void compute(NodeContext& ctx, stats::RNG& rng) const = 0;
  };

  /** @brief Factory function for label-based grouping. */
  Grouping::Ptr by_label();

  /** @brief Factory function for cutpoint-based grouping (regression). */
  Grouping::Ptr by_cutpoint();
}
