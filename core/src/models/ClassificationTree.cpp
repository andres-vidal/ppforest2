#include "models/ClassificationTree.hpp"

#include "utils/Invariant.hpp"

using namespace ppforest2::types;
using namespace ppforest2::stats;

namespace ppforest2 {
  using GroupIndices = ClassificationTree::GroupIndices;
  using Ptr          = ClassificationTree::Ptr;

  Ptr ClassificationTree::train(
      TrainingSpec const& s, FeatureMatrix& x, OutcomeVector& y, GroupPartition const& y_part, RNG& rng
  ) {
    invariant(is_classification(s), "ClassificationTree::train requires mode = Classification");

    TreeNode::Ptr root_ptr = Tree::build_root(s, x, y, y_part, rng);

    return std::make_unique<ClassificationTree>(std::move(root_ptr), TrainingSpec::make(s), y_part.groups);
  }

  FeatureVector ClassificationTree::predict(FeatureVector const& x, Proportions) const {
    return predict(x, Proportions{}, group_indices(groups));
  }

  FeatureVector ClassificationTree::predict(FeatureVector const& x, Proportions, GroupIndices const& indices) const {
    int const G                                         = static_cast<int>(indices.size());
    Outcome const pred                                  = Tree::predict(x);
    FeatureVector proportions                           = FeatureVector::Zero(G);
    proportions(indices.at(static_cast<GroupId>(pred))) = Feature(1);
    return proportions;
  }

  FeatureMatrix ClassificationTree::predict(FeatureMatrix const& x, Proportions) const {
    return predict(x, Proportions{}, group_indices(groups));
  }

  FeatureMatrix ClassificationTree::predict(FeatureMatrix const& x, Proportions, GroupIndices const& indices) const {
    int const G = static_cast<int>(indices.size());
    int const n = static_cast<int>(x.rows());
    FeatureMatrix proportions(n, G);

    for (int i = 0; i < n; ++i) {
      proportions.row(i) = predict(static_cast<FeatureVector>(x.row(i)), Proportions{}, indices);
    }

    return proportions;
  }

  void ClassificationTree::accept(Model::Visitor& visitor) const {
    visitor.visit(*this);
  }
}
