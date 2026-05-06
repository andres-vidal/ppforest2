#pragma once

#include "utils/Types.hpp"

#include "models/Bagged.hpp"
#include "models/Model.hpp"
#include "models/TreeNode.hpp"


namespace ppforest2 {
  /**
   * @brief Abstract base class for projection pursuit decision trees.
   *
   * Each internal node projects data onto a linear combination of
   * features and splits on the projected value. Leaf values depend on
   * the mode — group labels for classification, mean response for
   * regression — implemented in the concrete subclasses
   * `ClassificationTree` and `RegressionTree`.
   *
   * Construct via the static `Tree::train` factory, which dispatches
   * to the correct concrete type based on `training_spec.mode`.
   *
   * @code
   *   auto tree = Tree::train(spec, x, y);  // classification or regression
   *   Outcome label = tree->predict(x.row(0));
   *   OutcomeVector preds = tree->predict(x);
   * @endcode
   */
  class Tree : public Model {
  public:
    using Ptr = std::unique_ptr<Tree>;
    using Model::predict;

    using FeatureMatrix = types::FeatureMatrix;
    using FeatureVector = types::FeatureVector;
    using OutcomeVector = types::OutcomeVector;
    using Outcome       = types::Outcome;
    using RNG           = stats::RNG;

    using Root = TreeNode::Ptr;

    using GroupPartition = stats::GroupPartition;

    /** @brief Root node of the tree. */
    Root root;

    /**
     * @brief Train a tree from a response vector.
     *
     * Dispatches to `ClassificationTree::train` or `RegressionTree::train`
     * based on `training_spec.mode`. Creates an RNG from the spec's seed.
     *
     * `x` and `y` are taken by mutable reference because some strategies —
     * notably `ByCutpoint` for regression — permute rows in place during
     * training. Classification training does not mutate them, so the
     * classification path pays no cost for this signature. The alternative
     * (const-correct public API + defensive copy at the regression dispatch)
     * would force a full-matrix copy per single-tree regression call, which
     * is a real cost the library shouldn't absorb when the natural callers
     * (R bindings, CLI) discard the data right after training anyway.
     * Callers who need to preserve the original row order must copy before
     * calling.
     *
     * @param spec  Training specification (strategies + mode).
     * @param x     Feature matrix (n × p). May be permuted during regression training.
     * @param y     Response vector (n) — integer labels for classification,
     *              continuous response for regression. May be permuted during
     *              regression training.
     * @return      Trained tree as a polymorphic pointer.
     */
    static Ptr train(TrainingSpec const& spec, types::FeatureMatrix& x, types::OutcomeVector& y);

    /** @copydoc Tree::train(TrainingSpec const&, FeatureMatrix&, OutcomeVector&) */
    static Ptr train(TrainingSpec const& spec, types::FeatureMatrix& x, types::OutcomeVector& y, stats::RNG& rng);


    /**
     * @brief Predict a single observation.
     *
     * Walks the tree and returns the leaf value. Same implementation
     * for both modes — the leaf value is produced by the mode-specific
     * leaf strategy during training.
     */
    types::Outcome predict(types::FeatureVector const& x) const override;

    /** @brief Accept a model visitor (mode-specific dispatch). */
    void accept(Model::Visitor& visitor) const override = 0;

    bool operator==(Tree const& other) const;
    bool operator!=(Tree const& other) const;

  protected:
    Tree(TreeNode::Ptr root, TrainingSpec::Ptr spec)
        : root(std::move(root)) {
      training_spec = std::move(spec);
      degenerate    = this->root && this->root->degenerate;
    }

    /**
     * @brief Build the root node of a tree.
     *
     * Iteratively grows the tree from the given group partition. Shared
     * implementation used by `ClassificationTree::train` and
     * `RegressionTree::train`.
     *
     * `x` and `y` are mutable: regression's `ByCutpoint` grouping strategy
     * reorders rows in place. Classification doesn't mutate either — the
     * reference is still non-const to keep the signature uniform across
     * modes and avoid the "aliased mutable vs immutable view" smell of
     * the prior pointer-based design.
     *
     * @param spec       Training specification (strategies + mode).
     * @param x          Feature matrix. Mutated by regression's in-place
     *                   row reorder; untouched by classification.
     * @param y          Response vector. Same mutation contract as `x`.
     * @param y_part  Initial group partition for the root node.
     * @param rng        Random number generator (tree-local).
     * @return           Root `TreeNode` of the constructed tree.
     */
    static Root
    build_root(TrainingSpec const& spec, FeatureMatrix& x, OutcomeVector& y, GroupPartition const& y_part, RNG& rng);
  };

  /**
   * @brief Alias for the dominant `Bagged` instantiation in this codebase —
   * a bootstrap-aggregated `Tree`. Inner tree is polymorphic (classification
   * or regression via the `Tree` base); the wrapper itself is mode-agnostic.
   */
  using BaggedTree = Bagged<Tree>;
}
