#pragma once

#include "models/TreeNode.hpp"
#include "models/strategies/Strategy.hpp"
#include "stats/Stats.hpp"

namespace ppforest2 {
  struct NodeContext;
}

/**
 * @brief Leaf creation strategies.
 *
 * Controls what leaf node is produced when the tree stops splitting,
 * whether triggered by the stop rule or by a degenerate condition.
 * The default MajorityVote returns a TreeLeaf with the most
 * frequent class. Future strategies can return probability leaves,
 * regression leaves, etc.
 */
namespace ppforest2::leaf {
  /**
   * @brief Abstract strategy for creating leaf nodes.
   *
   * Returns a leaf node. Does not write to the context.
   */
  class LeafStrategy : public Strategy<LeafStrategy> {
  public:
    /**
     * @brief Create a leaf node for the given node context.
     *
     * Public NVI entry point that delegates to the subclass-provided `compute`.
     */
    TreeNode::Ptr create_leaf(NodeContext const& ctx, stats::RNG& rng) const { return compute(ctx, rng); }

  protected:
    /** @brief Subclass implementation of leaf construction. */
    virtual TreeNode::Ptr compute(NodeContext const& ctx, stats::RNG& rng) const = 0;
  };

  /** @brief Factory function for majority-vote leaf strategy. */
  LeafStrategy::Ptr majority_vote();

  /** @brief Factory function for mean-response leaf strategy. */
  LeafStrategy::Ptr mean_response();
}
