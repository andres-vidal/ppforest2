#pragma once

#include "models/Forest.hpp"
#include "stats/GroupPartition.hpp"

#include <map>
#include <set>

namespace ppforest2 {
  /**
   * @brief Random forest of classification trees.
   *
   * Aggregates per-tree predictions by majority vote. Use the free
   * functions in `Metrics.hpp` for OOB diagnostics and variable
   * importance, and `predict_proportions` (in `Model.hpp`) for
   * per-group vote proportions.
   */
  class ClassificationForest : public Forest {
  public:
    using Ptr = std::unique_ptr<ClassificationForest>;
    using Forest::predict;

    using FeatureMatrix = types::FeatureMatrix;
    using FeatureVector = types::FeatureVector;
    using OutcomeVector = types::OutcomeVector;
    using Outcome       = types::Outcome;
    using RNG           = stats::RNG;

    using Groups         = std::set<types::GroupId>;
    using GroupIndices   = std::map<types::GroupId, int>;
    using GroupPartition = stats::GroupPartition;


    /**
     * @brief Set of group labels this forest predicts over.
     *
     * Canonical column layout for `predict(FeatureMatrix, Proportions)`.
     * Populated at construction (training: from `y_part.groups`; JSON: from
     * `meta.groups` indices; R: from `forest$groups` names). Independent of
     * any individual tree's `node_groups()` — a degenerate-retry tree with
     * a narrower group set can't shrink the prediction matrix.
     */
    Groups groups;

    ClassificationForest(TrainingSpec::Ptr spec, Groups groups)
        : Forest(std::move(spec))
        , groups(std::move(groups)) {
      invariant(this->training_spec != nullptr, "ClassificationForest requires a non-null TrainingSpec");
      invariant(is_classification(this), "ClassificationForest requires a classification TrainingSpec");
      invariant(!this->groups.empty(), "ClassificationForest requires a non-empty groups set");
    }

    static Ptr train(TrainingSpec const& spec, FeatureMatrix const& x, OutcomeVector const& y);

    types::Outcome predict(FeatureVector const& x) const override;

    /** @brief Vote proportions for one observation, columns laid out by `groups`. */
    FeatureVector predict(FeatureVector const& x, Proportions) const;

    /**
     * @brief Vote proportions for one observation, with an explicit
     *        column layout passed as a precomputed `{group → column}` map.
     *
     * **Base primitive.** The matrix overload calls this once per row
     * with the same `indices`, so the map is built only once per
     * `predict(matrix, ...)` call rather than `n` times.
     */
    FeatureVector predict(FeatureVector const& x, Proportions, GroupIndices const& indices) const;

    /** @brief Vote proportions per row, columns laid out by `groups`. */
    FeatureMatrix predict(FeatureMatrix const& x, Proportions) const;

    /**
     * @brief Vote proportions per row, with an explicit column layout.
     *
     * Iterates rows and calls the single-row `predict(..., indices)`
     * overload — `indices` is built zero times here.
     */
    FeatureMatrix predict(FeatureMatrix const& x, Proportions, GroupIndices const& indices) const;

    void accept(Model::Visitor& visitor) const override;

  protected:
    BaggedTree::Ptr train_tree(FeatureMatrix const& x, OutcomeVector const& y, RNG& rng) const override;

  private:
    /**
     * @brief Transient pointer to the parent's label partition.
     *
     * Set by `train` to a stack-local in its own frame just before
     * `build_trees` runs, and cleared (via RAII) on return or unwind.
     * Read by `train_tree` under the OMP loop. Outside that one call
     * the pointer is `nullptr` — it never participates in the trained
     * forest's identity (no JSON, no equality, no predict).
     *
     * Pointer (not value) because `GroupPartition` has `const` members;
     * by-value storage would force `optional` plus move-construction
     * gymnastics, and we don't want to own it anyway.
     */
    GroupPartition const* y_part = nullptr;
  };
}
