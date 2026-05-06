#pragma once

#include "models/TreeNode.hpp"
#include "models/Projector.hpp"
#include "utils/Types.hpp"

namespace ppforest2 {
  /**
   * @brief Internal split node in a projection pursuit tree.
   *
   * Projects an observation onto @c projector and compares the result
   * against @c cutpoint.  If the projected value is below the cutpoint
   * the observation goes to the @c lower child; otherwise to @c upper.
   */
  class TreeBranch final : public TreeNode {
  public:
    using Ptr = std::unique_ptr<TreeBranch>;

    /** @brief Projection vector (p). Defines the linear combination of features. */
    pp::Projector projector;
    /** @brief Split cutpoint on the projected value. */
    types::Feature cutpoint;
    /** @brief Child node for observations with projected value < cutpoint. */
    TreeNode::Ptr lower;
    /** @brief Child node for observations with projected value ≥ cutpoint. */
    TreeNode::Ptr upper;

    /** @brief Set of group labels reachable from this node. */
    std::set<types::GroupId> groups;
    /** @brief Projection pursuit index value achieved at this split. */
    types::Feature pp_index_value = 0;

    TreeBranch(
        pp::Projector projector,
        types::Feature cutpoint,
        TreeNode::Ptr lower,
        TreeNode::Ptr upper,
        std::set<types::GroupId> groups = {},
        types::Feature pp_index_value   = 0
    );

    void accept(TreeNode::Visitor& visitor) const override;

    /** @brief Returns the response of the lower child. */
    types::Outcome response() const override;

    /**
     * @brief Route an observation through this split.
     *
     * Projects @p x onto the projector, compares against the
     * cutpoint, and delegates to the appropriate child node.
     *
     * @param x  Feature vector (p).
     * @return   Predicted group label from the reached leaf.
     */
    types::Outcome predict(types::FeatureVector const& x) const override;

    int group_count() const override { return static_cast<int>(groups.size()); }

    std::set<types::GroupId> node_groups() const override { return groups; }

    bool equals(TreeNode const& other) const override;
    TreeNode::Ptr clone() const override;

    /** @brief Factory method that returns a unique_ptr to a new TreeBranch. */
    static Ptr make(
        pp::Projector projector,
        types::Feature cutpoint,
        TreeNode::Ptr lower,
        TreeNode::Ptr upper,
        std::set<types::GroupId> groups = {},
        types::Feature pp_index_value   = 0
    );
  };
}
