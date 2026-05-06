#pragma once

#include "models/Bagged.hpp"
#include "models/Model.hpp"
#include "models/Tree.hpp"
#include "stats/Stats.hpp"

#include <memory>
#include <vector>

namespace ppforest2 {
  /**
   * @brief Abstract base class for projection pursuit random forests.
   *
   * Holds a vector of `BaggedTree` wrappers (each pairs a `Tree` with the
   * bootstrap sample indices it was trained on) and the shared training
   * spec. Aggregation logic (majority vote vs mean), proportion
   * predictions, and OOB handling are defined in the concrete subclasses
   * `ClassificationForest` and `RegressionForest`.
   *
   * Construct via `Forest::train`, which dispatches to the correct
   * concrete type based on `training_spec.mode`. Diagnostics and
   * variable importance are free functions in `models/Evaluation.hpp` —
   * they take a `Forest const&` (or `Ptr`) and dispatch on mode internally.
   *
   * @code
   *   auto forest = Forest::train(spec, x, y);
   *   auto preds  = forest->predict(x_test);
   * @endcode
   */
  class Forest : public Model {
  public:
    using Ptr = std::unique_ptr<Forest>;
    using Model::predict;

    using FeatureMatrix = types::FeatureMatrix;
    using FeatureVector = types::FeatureVector;
    using OutcomeVector = types::OutcomeVector;
    using Outcome       = types::Outcome;
    using RNG           = stats::RNG;

    /** @brief Bootstrap-aggregated trees. Each `BaggedTree` pairs the
     * polymorphic inner `Tree` with its sample indices. */
    std::vector<BaggedTree::Ptr> trees;

    /**
     * @brief Train a random forest.
     *
     * Dispatches to `ClassificationForest::train` or `RegressionForest::train`
     * based on `training_spec.mode`. Note that the top-level `x` / `y` are
     * not mutated here — each bootstrap tree resamples into its own local
     * storage. (Contrast with single-tree `Tree::train`, where regression
     * mode does permute `x` / `y` in place.)
     */
    static Ptr train(TrainingSpec const& spec, FeatureMatrix const& x, OutcomeVector const& y);

    /** @brief Per-row prediction (mode-specific: majority vote or mean). */
    Outcome predict(FeatureVector const& x) const override = 0;

    /** @brief Accept a model visitor (mode-specific dispatch). */
    void accept(Model::Visitor& visitor) const override = 0;

    /**
     * @brief Add a trained bagged tree to the forest.
     *
     * Asserts at runtime that the incoming tree's mode matches this
     * forest's mode (e.g., a `ClassificationForest` can only accept trees
     * trained under classification). This is the type-safety compromise
     * of keeping `Forest::trees` mode-agnostic at the container level:
     * the check runs at assembly time rather than at every prediction
     * site, but is cheaper than threading templates through every call
     * site that handles a `Forest::Ptr`.
     */
    void add_tree(BaggedTree::Ptr tree);

    bool operator==(Forest const& other) const;
    bool operator!=(Forest const& other) const;

  protected:
    Forest() = default;
    explicit Forest(TrainingSpec::Ptr spec) { training_spec = std::move(spec); };

    /**
     * @brief Train one bagged tree on a bootstrap resample of @p x / @p y.
     *
     * Mode-specific hook invoked by `build_trees` once per slot in the
     * forest (with retries on degenerate trees). Subclasses that need
     * per-training shared state (e.g. `ClassificationForest` caches the
     * parent's label partition for stratified sampling) hold it as a
     * private transient pointer set up before `build_trees` runs.
     *
     * Called from inside `build_trees`'s OpenMP parallel for, so
     * implementations must be thread-safe and read-only over `*this`.
     */
    virtual BaggedTree::Ptr train_tree(FeatureMatrix const& x, OutcomeVector const& y, RNG& rng) const = 0;

    /**
     * @brief Build `training_spec->size` bagged trees in parallel and
     *        attach them to this forest.
     *
     * Shared training-loop scaffolding for `ClassificationForest::train`
     * and `RegressionForest::train`. The two only differ in how a single
     * bagged tree is trained — the outer scaffolding (size guard, OpenMP
     * setup, per-tree retry loop with stream-id RNG, error capture, the
     * post-loop assembly with degenerate-flag propagation) is identical.
     * Mirrors the `Tree::build_root` pattern: shared algorithm lives on
     * the base; the concrete subclass provides the mode-specific work
     * via the `train_tree` virtual hook.
     *
     * Each tree is trained using a RNG stream formula `i + attempt * size`,
     * which is load-bearing for reproducibility.
     *
     * @throws Whatever `train_tree` threw. If multiple iterations threw,
     *         the first (lowest-index) exception is rethrown.
     */
    void build_trees(types::FeatureMatrix const& x, types::OutcomeVector const& y);
  };
}
