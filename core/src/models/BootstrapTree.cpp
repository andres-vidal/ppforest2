#include "models/BootstrapTree.hpp"
#include <algorithm>
#include <numeric>
#include <set>
#include <Eigen/Dense>

using namespace ppforest2::stats;
using namespace ppforest2::types;

namespace ppforest2 {
  BootstrapTree::Ptr BootstrapTree::train(
    TrainingSpec const&   training_spec,
    FeatureMatrix const&  x,
    GroupPartition const& group_spec,
    RNG &                 rng) {
    std::vector<int> sample_indices;
    sample_indices.reserve(x.rows());

    for (const auto& group : group_spec.groups) {
      const int group_size = group_spec.group_size(group);
      const int min_index  = group_spec.group_start(group);
      const int max_index  = group_spec.group_end(group);

      const Uniform unif(min_index, max_index);

      for (int j = 0; j < group_size; j++) {
        sample_indices.push_back(unif(rng));
      }
    }

    std::sort(sample_indices.begin(), sample_indices.end());

    FeatureMatrix sampled_x = x(sample_indices, Eigen::placeholders::all);

    Tree tree = Tree::train(training_spec, sampled_x, group_spec, rng);

    return std::make_unique<BootstrapTree>(
      std::move(tree.root),
      training_spec.clone(),
      std::move(sample_indices));
  }

  std::vector<int> BootstrapTree::oob_indices(int n_total) const {
    std::set<int> in_bag(sample_indices.begin(), sample_indices.end());
    std::vector<int> oob;
    oob.reserve(static_cast<std::size_t>(n_total) - in_bag.size());

    for (int i = 0; i < n_total; ++i) {
      if (in_bag.find(i) == in_bag.end()) {
        oob.push_back(i);
      }
    }

    return oob;
  }

  ResponseVector BootstrapTree::predict_oob(
    const FeatureMatrix&    x,
    const std::vector<int>& row_idx) const {
    if (row_idx.empty()) {
      return ResponseVector(0);
    }

    return predict(static_cast<FeatureMatrix>(x(row_idx, Eigen::placeholders::all)));
  }

  BootstrapTree::BootstrapTree(
    TreeNode::Ptr     root,
    TrainingSpec::Ptr spec,
    std::vector<int>  samp)  :
    Tree(std::move(root), std::move(spec)),
    sample_indices(std::move(samp)) {
  }
}
