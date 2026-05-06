#pragma once

#include "models/Tree.hpp"

#include <map>
#include <set>

namespace ppforest2 {
  /**
   * @brief A projection pursuit decision tree for classification.
   *
   * Leaves hold integer group labels produced by the configured leaf
   * strategy (default `MajorityVote`). The `predict(data, Proportions)`
   * overload returns a one-hot encoding of the predicted group.
   */
  class ClassificationTree : public Tree {
  public:
    using Ptr = std::unique_ptr<ClassificationTree>;
    using Tree::predict;

    using FeatureMatrix = types::FeatureMatrix;
    using FeatureVector = types::FeatureVector;
    using OutcomeVector = types::OutcomeVector;
    using Outcome       = types::Outcome;
    using RNG           = stats::RNG;

    using Groups         = std::set<types::GroupId>;
    using GroupIndices   = std::map<types::GroupId, int>;
    using GroupPartition = stats::GroupPartition;

    /**
     * @brief Set of group labels this tree predicts over.
     *
     * Canonical column layout for `predict(FeatureMatrix, Proportions)`.
     * Populated by `train` (from the parent's `y_part.groups`) and by
     * the JSON deserializer (from `meta.groups`). Independent of
     * `root->node_groups()` so that a stop/leaf combination producing a
     * single-leaf root doesn't shrink the prediction matrix relative to
     * what the model was trained for.
     */
    Groups groups;

    ClassificationTree(TreeNode::Ptr root, TrainingSpec::Ptr spec, Groups groups)
        : Tree(std::move(root), std::move(spec))
        , groups(std::move(groups)) {
      invariant(this->training_spec != nullptr, "ClassificationTree requires a non-null TrainingSpec");
      invariant(is_classification(this), "ClassificationTree requires a classification TrainingSpec");
      invariant(!this->groups.empty(), "ClassificationTree requires a non-empty groups set");
    }


    /**
     * @brief Train a classification tree with an external RNG.
     *
     * `x` and `y` are non-const for signature symmetry with
     * `RegressionTree::train` — classification doesn't mutate either.
     *
     * @param s    Training specification (must have `mode = Classification`).
     * @param x       Feature matrix (n × p).
     * @param y       Response vector (integer class labels encoded as floats).
     * @param y_part  Initial root group partition.
     * @param rng     Random number generator.
     */
    static Ptr train(TrainingSpec const& s, FeatureMatrix& x, OutcomeVector& y, GroupPartition const& y_part, RNG& rng);

    /**
     * @brief One-hot encoding of the predicted group for one observation,
     *        with columns laid out by `groups()`.
     */
    types::FeatureVector predict(types::FeatureVector const& x, Proportions) const;

    /**
     * @brief One-hot encoding for one observation, with an explicit column
     *        layout passed as a precomputed `{group → column}` map.
     *
     * **Base primitive.** `indices` must contain every label this tree can
     * predict (otherwise `at()` throws). The matrix overload, and
     * `ClassificationForest`, both build the indices once at entry and
     * call this overload per row so the map is never rebuilt in a loop.
     */
    FeatureVector predict(FeatureVector const& x, Proportions, GroupIndices const& indices) const;

    /**
     * @brief One-hot encoding of the predicted group per row, columns
     *        laid out by `groups()`.
     */
    FeatureMatrix predict(FeatureMatrix const& x, Proportions) const;

    /**
     * @brief One-hot encoding per row, with an explicit column layout.
     *
     * Iterates rows calling the single-row `predict(..., indices)` overload.
     * `ClassificationForest::predict(..., Proportions)` reuses this so
     * every bagged tree's proportions land in the forest's column layout.
     */
    FeatureMatrix predict(FeatureMatrix const& x, Proportions, GroupIndices const& indices) const;

    void accept(Model::Visitor& visitor) const override;
  };
}
