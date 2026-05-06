#include "models/TreeLeaf.hpp"
#include "models/TreeBranch.hpp"

#include <memory>

using namespace ppforest2::types;

namespace ppforest2 {
  TreeLeaf::TreeLeaf(Outcome value)
      : value(value) {}

  void TreeLeaf::accept(TreeNode::Visitor& visitor) const {
    visitor.visit(*this);
  }

  Outcome TreeLeaf::response() const {
    return value;
  }

  Outcome TreeLeaf::predict(FeatureVector const& /*x*/) const {
    return value;
  }

  bool TreeLeaf::equals(TreeNode const& other) const {
    struct EqualsVisitor : TreeNode::Visitor {
      Outcome value;
      bool result = false;

      explicit EqualsVisitor(Outcome v)
          : value(v) {}

      void visit(TreeLeaf const& leaf) override { result = (value == leaf.value); }
    };

    EqualsVisitor visitor(value);
    other.accept(visitor);
    return visitor.result;
  }

  TreeNode::Ptr TreeLeaf::clone() const {
    return std::make_unique<TreeLeaf>(*this);
  }

  TreeLeaf::Ptr TreeLeaf::make(Outcome value) {
    return std::make_unique<TreeLeaf>(value);
  }
}
