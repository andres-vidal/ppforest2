#pragma once

#include "models/TrainingSpec.hpp"
#include "utils/Types.hpp"

#include <memory>

namespace ppforest2 {
  class Tree;
  class Forest;
  class ClassificationTree;
  class ClassificationForest;
  class RegressionTree;
  class RegressionForest;

  /**
   * @brief Tag type for requesting vote-proportion predictions.
   *
   * Classification-only. `ClassificationTree::predict(data, Proportions{})`
   * returns a per-row one-hot of the predicted group; `ClassificationForest`
   * returns the per-tree vote proportions.
   */
  struct Proportions {};


  /**
   * @brief Abstract base class for predictive models (trees and forests).
   */
  class Model {
  public:
    using Ptr = std::shared_ptr<Model>;

    /**
     * @brief Visitor interface for model dispatch.
     *
     * Two layers of dispatch:
     *   - `visit(Tree)` / `visit(Forest)` — bimodal (classification or
     *     regression). Default to no-op so visitors can override only
     *     the cases they care about.
     *   - `visit(ClassificationTree)` / `visit(RegressionTree)` /
     *     `visit(ClassificationForest)` / `visit(RegressionForest)` —
     *     mode-specific. Default implementations delegate to the
     *     bimodal overload, so a visitor that overrides only `visit(Tree)`
     *     still receives both classification and regression trees.
     *
     * Visitors that don't care about mode override only the bimodal
     * pair. Visitors that need to call mode-specific API (e.g., the
     * classification-only `predict(data, Proportions{})`) override the
     * relevant mode-specific overload(s).
     */
    class Visitor {
    public:
      virtual ~Visitor() = default;

      virtual void visit(Tree const&) {}
      virtual void visit(Forest const&) {}

      virtual void visit(ClassificationTree const& tree);
      virtual void visit(ClassificationForest const& forest);
      virtual void visit(RegressionTree const& tree);
      virtual void visit(RegressionForest const& forest);
    };

    virtual ~Model() = default;

    /** @brief Whether the model contains degenerate nodes/splits. */
    bool degenerate = false;

    /** @brief Training specification used to build this model. */
    TrainingSpec::Ptr training_spec;

    /** @brief Accept a model visitor (double dispatch). */
    virtual void accept(Visitor& visitor) const = 0;

    /**
     * @brief Predict a single observation.
     *
     * @param x  Feature vector (p).
     * @return   Predicted group label.
     */
    virtual types::Outcome predict(types::FeatureVector const& x) const = 0;

    /**
     * @brief Predict a matrix of observations.
     *
     * Default implementation iterates rows and dispatches to the
     * single-row `predict`. Subclasses may override to vectorize.
     *
     * @param x  Feature matrix (n × p).
     * @return   Predicted group labels (n).
     */
    virtual types::OutcomeVector predict(types::FeatureMatrix const& x) const {
      types::OutcomeVector result(x.rows());

      for (Eigen::Index i = 0; i < x.rows(); ++i) {
        result(i) = predict(static_cast<types::FeatureVector>(x.row(i)));
      }

      return result;
    }

    /**
     * @brief Train a model from a training specification.
     *
     * Dispatches to Tree::train or Forest::train based on
     * spec.is_forest().
     *
     * @param spec  Training specification.
     * @param x     Feature matrix (n × p).
     * @param y     Outcome vector (n).
     * @return      Trained model (Tree or Forest).
     */
    static Ptr train(TrainingSpec const& spec, types::FeatureMatrix& x, types::OutcomeVector& y);

    /**
     * @brief Validate common training inputs (y non-empty, matching x rows).
     *
     * Throws `UserError` on failure. Called by `Tree::train` and `Forest::train`
     * at their entry points so both modes get the same validation and the
     * checks aren't duplicated per-mode.
     */
    static void check_train_inputs(types::FeatureMatrix const& x, types::OutcomeVector const& y);
  };

  /**
   * @brief Compute vote proportions for a classification model.
   *
   * Routes through `Model::Visitor` to `ClassificationTree::predict(x, Proportions{})`
   * or `ClassificationForest::predict(x, Proportions{})`. Throws `UserError`
   * if the model is a regression tree or forest. Centralises the visitor
   * boilerplate that the CLI, R bindings, and golden-gen would otherwise
   * each repeat.
   */
  types::FeatureMatrix predict_proportions(Model const& model, types::FeatureMatrix const& x);

  /** @brief Single-row vote proportions for a classification model. */
  types::FeatureVector predict_proportions(Model const& model, types::FeatureVector const& x);

  /**
   * @brief Whether @p model was trained for classification.
   *
   * Encapsulates the `training_spec` null-check that callers would
   * otherwise repeat. Returns `false` for an unconfigured model.
   */
  inline bool is_classification(Model const& model) {
    return is_classification(model.training_spec);
  }

  /**
   * @brief Whether @p model was trained for regression.
   *
   * Encapsulates the `training_spec` null-check that callers would
   * otherwise repeat. Returns `false` for an unconfigured model.
   */
  inline bool is_regression(Model const& model) {
    return is_regression(model.training_spec);
  }

  /** @brief Pointer overload — null-safe; returns `false` for a null model. */
  inline bool is_classification(Model const* model) {
    return model != nullptr && is_classification(*model);
  }

  /** @brief Pointer overload — null-safe; returns `false` for a null model. */
  inline bool is_regression(Model const* model) {
    return model != nullptr && is_regression(*model);
  }
}
