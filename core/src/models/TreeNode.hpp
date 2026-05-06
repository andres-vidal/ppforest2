#pragma once

#include <memory>
#include <set>

#include "utils/Types.hpp"

namespace ppforest2 {
  class TreeBranch;
  class TreeLeaf;

  /**
   * @brief Abstract base class for nodes in a projection pursuit tree.
   *
   * A tree is a recursive structure of nodes: internal split nodes
   * (TreeBranch) that project and split, and leaf nodes
   * (TreeLeaf) that hold a group label.
   */
  class TreeNode {
  public:
    using Ptr = std::unique_ptr<TreeNode>;

    /**
     * @brief Visitor interface for tree node dispatch.
     *
     * Implements the visitor pattern to distinguish between internal
     * split nodes (TreeBranch) and leaf nodes (TreeLeaf). Both visit
     * overloads default to no-op so visitors can override only the
     * node type they care about.
     */
    class Visitor {
    public:
      virtual ~Visitor() = default;

      virtual void visit(TreeBranch const&) {}
      virtual void visit(TreeLeaf const&) {}
    };

    /** @brief Whether this node (or any descendant) had a degenerate split. */
    bool degenerate = false;

    virtual ~TreeNode() = default;

    /** @brief Accept a tree node visitor (double dispatch). */
    virtual void accept(Visitor& visitor) const = 0;

    /**
     * @brief Predict the group label for a single observation.
     *
     * @param x  Feature vector (p).
     * @return   Predicted group label.
     */
    virtual types::Outcome predict(types::FeatureVector const& x) const = 0;

    /** @brief The group label at this node (leaf value or majority group). */
    virtual types::Outcome response() const = 0;

    /**
     * @brief Number of distinct groups reachable from this node.
     */
    virtual int group_count() const = 0;

    /**
     * @brief Sorted set of group labels reachable from this node.
     */
    virtual std::set<types::GroupId> node_groups() const = 0;

    /** @brief Structural equality comparison (value-based). */
    virtual bool equals(TreeNode const& other) const = 0;

    /** @brief Deep copy of this node and its subtree. */
    virtual Ptr clone() const = 0;

    bool operator==(TreeNode const& other) const;
    bool operator!=(TreeNode const& other) const;
  };

  /**
   * @brief Whether @p node is a `TreeLeaf`.
   *
   * Routes through `TreeNode::Visitor` rather than a virtual method on
   * `TreeNode` itself — keeps the base class focused on data/traversal
   * and matches the visitor-based dispatch used elsewhere in the codebase.
   * The `Ptr` overload is a thin dereferencing wrapper.
   */
  bool is_leaf(TreeNode const& node);
  bool is_leaf(TreeNode::Ptr const& node);
}
