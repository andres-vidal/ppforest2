#pragma once

#include "stats/Metrics.hpp"
#include "utils/Types.hpp"

#include <memory>
#include <optional>

/**
 * @file Evaluation.hpp
 * @brief Free-function API for evaluating trained models — variable
 * importance, out-of-bag diagnostics, and prediction error.
 *
 * Models (`Tree`, `Forest` and their concrete subclasses) own structure
 * and prediction; everything else lives here as free functions that take
 * a `Model const&`. Mode-specific dispatch goes through `Model::Visitor`,
 * so callers don't need to know whether a forest is classification or
 * regression — wrong-mode inputs raise `UserError`.
 *
 * Mirrors the pattern already used by `predict_proportions` (in
 * `Model.hpp`) and `is_leaf` (in `TreeNode.hpp`).
 */
namespace ppforest2 {
  class Model;
  class Tree;
  class Forest;
  class ClassificationForest;
  class RegressionForest;

  /**
   * @brief Grouped result of the variable importance measures.
   *
   * Produced by the `variable_importance(...)` overloads below.
   * `variable_importance(Tree)` fills `projections` and `scale`;
   * `variable_importance(Forest)` fills all four fields. Individual
   * measures can be computed via `vi_projections`, `vi_permuted`, and
   * `vi_weighted_projections` directly.
   */
  struct VariableImportance {
    /** @brief VI1 — per-variable permuted importance (forest only). */
    types::FeatureVector permuted;
    /** @brief VI2 — per-variable projection-coefficient importance. */
    types::FeatureVector projections;
    /** @brief VI3 — per-variable weighted-projection importance (forest only). */
    types::FeatureVector weighted_projections;
    /** @brief Per-variable σ used to rescale coefficients (columnwise sd). */
    types::FeatureVector scale;
  };

  // ---------------------------------------------------------------------------
  // Variable importance
  // ---------------------------------------------------------------------------

  /**
   * @brief VI2 for a single tree — projection-coefficient importance.
   *
   * Mode-agnostic (depends only on the tree's projector geometry).
   */
  types::FeatureVector vi_projections(Tree const& tree, int n_vars, types::FeatureVector const* scale = nullptr);

  /** @brief VI2 for a forest — averaged over non-degenerate trees. */
  types::FeatureVector vi_projections(Forest const& forest, int n_vars, types::FeatureVector const* scale = nullptr);

  /**
   * @brief VI1 — per-variable permuted importance.
   *
   * Classification: accuracy drop on permuted OOB rows. Regression:
   * NMSE increase. Throws `UserError` if @p forest is not a recognised mode.
   */
  types::FeatureVector
  vi_permuted(Forest const& forest, types::FeatureMatrix const& x, types::OutcomeVector const& y, int seed);

  /**
   * @brief VI3 — weighted projection-coefficient importance.
   *
   * Each tree's contribution is weighted by a per-tree OOB quality score
   * (mode-specific). Throws `UserError` for unknown forest modes.
   */
  types::FeatureVector vi_weighted_projections(
      Forest const& forest,
      types::FeatureMatrix const& x,
      types::OutcomeVector const& y,
      types::FeatureVector const* scale = nullptr
  );

  /** @brief Bundle the available VI measures for a single tree (VI2 only). */
  VariableImportance variable_importance(Tree const& tree, types::FeatureMatrix const& x);

  /** @brief Bundle all three VI measures for a forest. */
  VariableImportance
  variable_importance(Forest const& forest, types::FeatureMatrix const& x, types::OutcomeVector const& y, int seed);

  // ---------------------------------------------------------------------------
  // Out-of-bag diagnostics
  // ---------------------------------------------------------------------------

  /**
   * @brief Out-of-bag predictions.
   *
   * Classification: majority-vote labels. Regression: mean of OOB tree
   * predictions. Rows that no tree left out-of-bag are filled with
   * `NaN` (the same sentinel for both modes — see `oob_predict`'s
   * implementation comment for the rationale). Throws `UserError` for
   * unknown modes.
   *
   * Prefer `oob_metrics` when you want diagnostics — it returns the
   * filtered metric directly with no sentinel exposed to the caller.
   */
  types::OutcomeVector oob_predict(Forest const& forest, types::FeatureMatrix const& x);

  /**
   * @brief Out-of-bag metrics — sentinel-free summary of OOB performance.
   *
   * Mode-specific overloads: classification returns `ClassificationMetrics`
   * (confusion matrix + error rate), regression returns `RegressionMetrics`
   * (MSE / MAE / R²). Both return `std::nullopt` when no observation has any
   * OOB tree (which only happens for empty forests in practice). The internal
   * "no OOB" sentinel never leaks to the caller — these functions own the
   * sentinel-filter logic.
   */
  stats::ClassificationMetrics::Maybe
  oob_metrics(ClassificationForest const& forest, types::FeatureMatrix const& x, types::OutcomeVector const& y);

  stats::RegressionMetrics::Maybe
  oob_metrics(RegressionForest const& forest, types::FeatureMatrix const& x, types::OutcomeVector const& y);

  /**
   * @brief Out-of-bag error.
   *
   * Classification: misclassification rate. Regression: mean squared error.
   * Returns `std::nullopt` if no observation has any OOB tree.
   * Throws `UserError` for unknown modes.
   */
  std::optional<double> oob_error(Forest const& forest, types::FeatureMatrix const& x, types::OutcomeVector const& y);

  /** @brief Convenience overload — accepts integer class labels for classification. */
  std::optional<double> oob_error(Forest const& forest, types::FeatureMatrix const& x, types::GroupIdVector const& y);

  // ---------------------------------------------------------------------------
  // Prediction error
  // ---------------------------------------------------------------------------

  /**
   * @brief Prediction error of @p model on data `(x, y)`.
   *
   * Classification: misclassification rate. Regression: mean squared error.
   * Mode dispatch goes through `Model::Visitor` so callers don't need to
   * know whether @p model is classification or regression.
   * Throws `UserError` for unknown modes.
   */
  double error(Model const& model, types::FeatureMatrix const& x, types::OutcomeVector const& y);

  // ---------------------------------------------------------------------------
  // Pointer-friendly overloads
  //
  // Each forwards to the reference overload via `*ptr`, so callers holding
  // a `Tree::Ptr` / `Forest::Ptr` (or any concrete subclass smart pointer
  // like `ClassificationForest::Ptr`) skip the dereference at the call site.
  //
  // Templated on `std::unique_ptr<T>` rather than overloaded per concrete
  // type so they cover unique_ptr<Tree>, unique_ptr<Forest>,
  // unique_ptr<ClassificationForest>, etc., uniformly. The `*` dereference
  // gives a `T const&` that resolves the right reference overload above.
  // ---------------------------------------------------------------------------

  template<typename T>
  types::FeatureVector
  vi_projections(std::unique_ptr<T> const& m, int n_vars, types::FeatureVector const* scale = nullptr) {
    return vi_projections(*m, n_vars, scale);
  }

  template<typename T>
  types::FeatureVector
  vi_permuted(std::unique_ptr<T> const& m, types::FeatureMatrix const& x, types::OutcomeVector const& y, int seed) {
    return vi_permuted(*m, x, y, seed);
  }

  template<typename T>
  types::FeatureVector vi_weighted_projections(
      std::unique_ptr<T> const& m,
      types::FeatureMatrix const& x,
      types::OutcomeVector const& y,
      types::FeatureVector const* scale = nullptr
  ) {
    return vi_weighted_projections(*m, x, y, scale);
  }

  template<typename T>
  VariableImportance variable_importance(std::unique_ptr<T> const& m, types::FeatureMatrix const& x) {
    return variable_importance(*m, x);
  }

  template<typename T>
  VariableImportance variable_importance(
      std::unique_ptr<T> const& m, types::FeatureMatrix const& x, types::OutcomeVector const& y, int seed
  ) {
    return variable_importance(*m, x, y, seed);
  }

  template<typename T> types::OutcomeVector oob_predict(std::unique_ptr<T> const& m, types::FeatureMatrix const& x) {
    return oob_predict(*m, x);
  }

  template<typename T>
  std::optional<double>
  oob_error(std::unique_ptr<T> const& m, types::FeatureMatrix const& x, types::OutcomeVector const& y) {
    return oob_error(*m, x, y);
  }

  template<typename T>
  std::optional<double>
  oob_error(std::unique_ptr<T> const& m, types::FeatureMatrix const& x, types::GroupIdVector const& y) {
    return oob_error(*m, x, y);
  }

  template<typename T>
  double error(std::unique_ptr<T> const& m, types::FeatureMatrix const& x, types::OutcomeVector const& y) {
    return error(*m, x, y);
  }
}
