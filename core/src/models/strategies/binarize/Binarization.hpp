#pragma once

#include "models/strategies/Strategy.hpp"
#include "stats/Stats.hpp"

/**
 * @brief Binarization strategies for multiclass-to-binary reduction.
 *
 * When a node has more than 2 groups, a binarization strategy reduces
 * it to a binary problem. The built-in LargestGap finds the largest
 * gap between sorted projected group means. Future strategies (e.g.
 * closest-pair from da Silva Extension I) can be plugged in.
 */
namespace ppforest2 {
  struct NodeContext;
}

namespace ppforest2::binarize {
  /**
   * @brief Abstract strategy for multiclass-to-binary reduction.
   *
   * Writes `ctx.y_bin`.
   */
  class Binarization : public Strategy<Binarization> {
  public:
    /**
     * @brief Reduce a multiclass partition to binary and store in context.
     *
     * Public NVI entry point: skips if `ctx.aborted` is set, otherwise
     * delegates to the subclass-provided `compute`.
     */
    void regroup(NodeContext& ctx, stats::RNG& rng) const;

  protected:
    /** @brief Subclass implementation of binarization. */
    virtual void compute(NodeContext& ctx, stats::RNG& rng) const = 0;
  };

  /** @brief Factory function for largest-gap binarization. */
  Binarization::Ptr largest_gap();

  /** @brief Factory function for the Disabled (placeholder) binarizer. */
  Binarization::Ptr disabled();
}
