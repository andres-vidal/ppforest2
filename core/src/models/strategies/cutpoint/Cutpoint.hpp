#pragma once

#include "models/strategies/Strategy.hpp"
#include "stats/Stats.hpp"

/**
 * @brief Cutpoint strategies for computing decision cutpoints.
 *
 * Contains the abstract Cutpoint interface and concrete implementations
 * that determine the split cutpoint in projected space. The built-in
 * MeanOfMeans uses the midpoint of the two group means.
 */
namespace ppforest2 {
  struct NodeContext;
}

namespace ppforest2::cutpoint {
  /**
   * @brief Abstract strategy for computing the split cutpoint.
   *
   * Writes `ctx.cutpoint`.
   */
  class Cutpoint : public Strategy<Cutpoint> {
  public:
    /**
     * @brief Locate the split cutpoint and store it in the context.
     *
     * Public NVI entry point: skips if `ctx.aborted` is set, otherwise
     * delegates to the subclass-provided `compute`.
     */
    void locate(NodeContext& ctx, stats::RNG& rng) const;

  protected:
    /** @brief Subclass implementation of cutpoint computation. */
    virtual void compute(NodeContext& ctx, stats::RNG& rng) const = 0;
  };

  /** @brief Factory function for mean-of-means split cutpoint. */
  Cutpoint::Ptr mean_of_means();
}
