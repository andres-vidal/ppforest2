#include "models/RegressionTree.hpp"

#include "utils/Invariant.hpp"

using namespace ppforest2::types;
using namespace ppforest2::stats;

namespace ppforest2 {
  RegressionTree::Ptr RegressionTree::train(
      TrainingSpec const& s, FeatureMatrix& x, OutcomeVector& y, GroupPartition const& y_part, RNG& rng
  ) {
    invariant(is_regression(s), "RegressionTree::train requires mode = Regression");

    TreeNode::Ptr root_ptr = Tree::build_root(s, x, y, y_part, rng);

    return std::make_unique<RegressionTree>(std::move(root_ptr), TrainingSpec::make(s));
  }

  void RegressionTree::accept(Model::Visitor& visitor) const {
    visitor.visit(*this);
  }
}
