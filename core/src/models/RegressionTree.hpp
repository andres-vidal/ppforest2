#pragma once

#include "models/Tree.hpp"

namespace ppforest2 {
  /**
   * @brief A projection pursuit decision tree for regression.
   *
   * Leaves hold continuous mean response values produced by the
   * `MeanResponse` leaf strategy. Training requires a `y`
   * vector; in-place reordering of feature rows happens inside the
   * build loop via the `ByCutpoint` grouping strategy.
   *
   */
  class RegressionTree : public Tree {
  public:
    using Ptr = std::unique_ptr<RegressionTree>;
    using Tree::predict;

    using FeatureMatrix = types::FeatureMatrix;
    using FeatureVector = types::FeatureVector;
    using OutcomeVector = types::OutcomeVector;
    using Outcome       = types::Outcome;
    using RNG           = stats::RNG;

    using GroupPartition = stats::GroupPartition;

    RegressionTree(TreeNode::Ptr root, TrainingSpec::Ptr spec)
        : Tree(std::move(root), std::move(spec)) {
      invariant(this->training_spec != nullptr, "RegressionTree requires a non-null TrainingSpec");
      invariant(is_regression(this), "RegressionTree requires a regression TrainingSpec");
    }

    /**
     * @brief Train a regression tree with an external RNG.
     *
     * Takes `x` and `y` by **non-const reference**. The
     * `ByCutpoint` grouping strategy reorders rows in place on the
     * caller's storage — there is no internal copy. Callers must pass
     * buffers they own and are willing to see mutated. Typical callers:
     *
     * - **Bootstrap trees**: pass freshly-built local subsamples. Zero
     *   additional copies beyond the subsample itself.
     * - **Single-tree `Tree::train` path**: the top-level dispatcher
     *   holds the caller's data as `const&`, so it makes a single copy
     *   of `x` and `y` at the call site before invoking this
     *   function. The copy is visible at the caller, not hidden inside.
     *
     * @param s    Training specification (must have `mode = Regression`).
     * @param x       Feature matrix (n × p), sorted by continuous response.
     *                Will be permuted in place during training.
     * @param y       Continuous response vector (n), same order as `x`.
     *                Will be permuted in place during training.
     * @param y_part  Initial root group partition (typically a median split).
     * @param rng     Random number generator.
     */
    static Ptr train(TrainingSpec const& s, FeatureMatrix& x, OutcomeVector& y, GroupPartition const& y_part, RNG& rng);

    void accept(Model::Visitor& visitor) const override;
  };
}
