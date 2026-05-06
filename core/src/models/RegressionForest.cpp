#include "models/RegressionForest.hpp"

#include "models/Bagged.hpp"
#include "models/RegressionTree.hpp"
#include "stats/Stats.hpp"
#include "stats/Uniform.hpp"
#include "utils/Invariant.hpp"

#include <Eigen/Dense>
#include <algorithm>

using namespace ppforest2::stats;
using namespace ppforest2::types;

namespace ppforest2 {
  namespace {
    /** @brief Uniform sample with replacement: n indices drawn from [0, n-1]. */
    std::vector<int> uniform_sample(int n, RNG& rng) {
      std::vector<int> indices;
      indices.reserve(n);

      Uniform const unif(0, n - 1);

      for (int j = 0; j < n; ++j) {
        indices.push_back(unif(rng));
      }

      std::sort(indices.begin(), indices.end());
      return indices;
    }
  }

  RegressionForest::RegressionForest() = default;

  RegressionForest::RegressionForest(TrainingSpec::Ptr spec)
      : Forest(std::move(spec)) {
    invariant(this->training_spec != nullptr, "RegressionForest requires a non-null TrainingSpec");
    invariant(is_regression(this), "RegressionForest requires a regression TrainingSpec");
  }

  RegressionForest::Ptr
  RegressionForest::train(TrainingSpec const& spec, FeatureMatrix const& x, OutcomeVector const& y) {
    invariant(is_regression(spec), "RegressionForest::train requires mode = Regression");

    auto forest = std::make_unique<RegressionForest>(TrainingSpec::make(spec));
    forest->build_trees(x, y);
    return forest;
  }

  BaggedTree::Ptr RegressionForest::train_tree(FeatureMatrix const& x, OutcomeVector const& y, RNG& rng) const {
    invariant(y.size() == x.rows(), "Response size must match x.rows() for regression");

    // Bootstrap indices come back ascending from `uniform_sample`, and the
    // training pipeline guarantees `x` rows are y-sorted (CSV reader sorts;
    // simulator emits in y-order). So `x(sample_indices, all)` is already
    // y-sorted — no extra sort needed for `ByCutpoint::compute_init`'s
    // contiguous-block contract.
    //
    // The `y_part` is rebuilt per bootstrap because each replicate has its
    // own median (contrast with ClassificationForest, which precomputes one
    // forest-level `y_part`).
    std::vector<int> sample_indices = uniform_sample(static_cast<int>(x.rows()), rng);
    FeatureMatrix sampled_x         = x(sample_indices, Eigen::all);
    OutcomeVector sampled_y         = y(sample_indices, Eigen::all).eval();

    GroupPartition sampled_gp = this->training_spec->init_groups(sampled_y);

    RegressionTree::Ptr tree = RegressionTree::train(*this->training_spec, sampled_x, sampled_y, sampled_gp, rng);

    return BaggedTree::make(std::move(tree), std::move(sample_indices));
  }

  Outcome RegressionForest::predict(FeatureVector const& x) const {
    invariant(!trees.empty(), "Forest has no trees.");

    std::vector<Outcome> preds;
    preds.reserve(trees.size());

    for (auto const& tree : trees) {
      preds.push_back(tree->predict(x));
    }

    return stats::mean(preds);
  }

  void RegressionForest::accept(Model::Visitor& visitor) const {
    visitor.visit(*this);
  }
}
