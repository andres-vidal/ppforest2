#pragma once

#include "models/strategies/Strategy.hpp"
#include "stats/Stats.hpp"

#include <vector>

/**
 * @brief Stop rule strategies that determine when to create leaf nodes.
 *
 * Controls tree growth by deciding when a node should stop splitting
 * and become a leaf. The built-in PureNode stops when only one group
 * remains. Future strategies may add max-depth or min-samples rules.
 */
namespace ppforest2 {
  struct NodeContext;
}

namespace ppforest2::stop {
  /**
   * @brief Abstract strategy for tree stopping rules.
   *
   * Returns true when the node should become a leaf. Does not write to
   * the context.
   */
  class StopRule : public Strategy<StopRule> {
  public:
    /**
     * @brief Determine whether to stop growing at this node.
     *
     * Public NVI entry point. Returns true if the configured stop rule
     * fires, OR if the node has fewer than 2 groups (a tree-level
     * invariant — no grouping strategy can split a single group, so
     * stopping is the only sane option). Otherwise delegates to the
     * subclass-provided `compute`.
     */
    bool should_stop(NodeContext const& ctx, stats::RNG& rng) const;

  protected:
    /** @brief Subclass implementation of the stop predicate. */
    virtual bool compute(NodeContext const& ctx, stats::RNG& rng) const = 0;
  };

  /** @brief Factory function for pure-node stop rule. */
  StopRule::Ptr pure_node();

  /** @brief Factory function for minimum-size stop rule. */
  StopRule::Ptr min_size(int n);

  /** @brief Factory function for minimum-variance stop rule. */
  StopRule::Ptr min_variance(types::Feature threshold);

  /** @brief Factory function for max-depth stop rule. */
  StopRule::Ptr max_depth(int n);

  /** @brief Factory function for composite stop rule (logical OR). */
  StopRule::Ptr any(std::vector<StopRule::Ptr> rules);
}
