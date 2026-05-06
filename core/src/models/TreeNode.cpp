
#include "models/TreeNode.hpp"
#include "models/TreeLeaf.hpp"

namespace ppforest2 {
  bool TreeNode::operator==(TreeNode const& other) const {
    return this->equals(other);
  }

  bool TreeNode::operator!=(TreeNode const& other) const {
    return !this->equals(other);
  }

  bool is_leaf(TreeNode const& node) {
    struct LeafCheck : TreeNode::Visitor {
      bool result = false;
      void visit(TreeLeaf const& /*node*/) override { result = true; }
    };

    LeafCheck visitor;
    node.accept(visitor);
    return visitor.result;
  }

  bool is_leaf(TreeNode::Ptr const& node) {
    return is_leaf(*node);
  }
}
