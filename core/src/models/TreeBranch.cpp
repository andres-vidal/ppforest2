#include "models/TreeBranch.hpp"
#include "utils/Math.hpp"

#include <stdexcept>
#include <utility>

using namespace ppforest2::types;
using namespace ppforest2::pp;

namespace ppforest2 {
  TreeBranch::TreeBranch(
      Projector projector,
      Feature cutpoint,
      TreeNode::Ptr lower,
      TreeNode::Ptr upper,
      std::set<GroupId> groups,
      Feature pp_index_value
  )
      : projector(std::move(projector))
      , cutpoint(cutpoint)
      , lower(std::move(lower))
      , upper(std::move(upper))
      , groups(std::move(groups))
      , pp_index_value(pp_index_value) {
    degenerate = this->lower->degenerate || this->upper->degenerate;
  }

  void TreeBranch::accept(TreeNode::Visitor& visitor) const {
    visitor.visit(*this);
  }

  Outcome TreeBranch::response() const {
    throw std::invalid_argument("Cannot get response from a condition node");
  }

  Outcome TreeBranch::predict(FeatureVector const& x) const {
    Feature const projected = x.dot(projector);

    if (projected < cutpoint) {
      return lower->predict(x);
    }

    return upper->predict(x);
  }

  bool TreeBranch::equals(TreeNode const& other) const {
    struct EqualsVisitor : TreeNode::Visitor {
      TreeBranch const& self;
      bool result = false;

      explicit EqualsVisitor(TreeBranch const& self)
          : self(self) {}

      void visit(TreeBranch const& branch) override {
        bool projectors_collinear = math::collinear(self.projector, branch.projector);
        bool cutpoint_approximate = math::is_approx(self.cutpoint, branch.cutpoint);

        bool lower_equal = *self.lower == *branch.lower;
        bool upper_equal = *self.upper == *branch.upper;

        result = projectors_collinear && cutpoint_approximate && lower_equal && upper_equal;
      }
    };

    EqualsVisitor visitor(*this);
    other.accept(visitor);
    return visitor.result;
  }

  TreeNode::Ptr TreeBranch::clone() const {
    return make(projector, cutpoint, lower->clone(), upper->clone(), groups, pp_index_value);
  }

  TreeBranch::Ptr TreeBranch::make(
      Projector projector,
      Feature cutpoint,
      TreeNode::Ptr lower,
      TreeNode::Ptr upper,
      std::set<GroupId> groups,
      Feature pp_index_value
  ) {
    return std::make_unique<TreeBranch>(
        std::move(projector), cutpoint, std::move(lower), std::move(upper), std::move(groups), pp_index_value
    );
  }
}
