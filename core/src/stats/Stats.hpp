#pragma once

#include "utils/RangeVector.hpp"
#include "utils/Types.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>
#include <Eigen/Dense>
#include <pcg_random.hpp>

/**
 * @brief Statistical infrastructure for training and evaluation.
 *
 * Provides the random number generator (pcg32), discrete uniform
 * sampling (Lemire's method), grouped-observation bookkeeping
 * (GroupPartition), confusion matrices, data simulation, and basic
 * descriptive statistics used throughout the training pipeline.
 */
namespace ppforest2::stats {
  using RNG = pcg32;


  /**
   * @brief Sort a feature matrix and a response vector by the response values.
   *
   * Templated on the response vector type so callers can sort by integer
   * group labels (`GroupIdVector`) or continuous responses (`OutcomeVector`).
   */
  template<typename Y> void sort(types::FeatureMatrix& x, Y& y) {
    std::vector<int> indices = utils::range_vector(x.rows());

    std::stable_sort(indices.begin(), indices.end(), [&y](int i, int j) { return y(i) < y(j); });

    x = x(indices, Eigen::all).eval();
    y = y(indices, Eigen::all).eval();
  }

  /**
   * @brief Unique group labels in a response vector.
   *
   * @param column  Group ID vector.
   * @return        Set of unique group labels.
   */
  std::set<types::GroupId> unique(types::GroupIdVector const& column);

  /**
   * @brief Sample variance of a vector (unbiased, `n-1` denominator).
   *
   * Returns 0 for single-element inputs (variance undefined; 0 is the
   * natural degenerate value for "no spread"). Throws on empty input.
   */
  template<typename Derived> double var(Eigen::MatrixBase<Derived> const& data) {
    static_assert(
        Derived::ColsAtCompileTime == 1 || Derived::ColsAtCompileTime == Eigen::Dynamic,
        "var: expected a vector (single column)"
    );

    if (data.rows() == 0) {
      throw std::invalid_argument("var: data must have at least one row");
    }

    if (data.rows() == 1) {
      return 0.0;
    }

    double const mean = static_cast<double>(data.mean());
    return (data.array().template cast<double>() - mean).square().sum() / static_cast<double>(data.rows() - 1);
  }

  /** @brief Sample standard deviation of a vector — `sqrt(var(data))`. */
  template<typename Derived> double sd(Eigen::MatrixBase<Derived> const& data) {
    return std::sqrt(var(data));
  }

  /**
   * @brief Column-wise sample variance of a matrix.
   *
   * @param data  Feature matrix with at least 2 rows.
   * @return      FeatureVector of size p (one σ² per column).
   */
  types::FeatureVector var(types::FeatureMatrix const& data);

  /** @brief Column-wise sample standard deviation — element-wise `sqrt` of `var`. */
  types::FeatureVector sd(types::FeatureMatrix const& data);

  /**
   * @brief Majority vote over a sequence of integer-coded class labels.
   *
   * Returns the label with the largest count. Ties are broken by the
   * smallest `GroupId` — `std::map` iterates in ascending key order and
   * the strict `>` comparison keeps the earliest winner.
   *
   * Used for classification ensemble aggregation: per-row OOB voting
   * (`models/Evaluation.cpp`) and per-row in-bag voting
   * (`ClassificationForest::predict`). Throws on empty input.
   */
  types::Outcome majority_vote(std::vector<types::Outcome> const& preds);

  /**
   * @brief Arithmetic mean of a sequence of outcome values.
   *
   * Used for regression ensemble aggregation: per-row OOB averaging
   * (`models/Evaluation.cpp`) and per-row in-bag averaging
   * (`RegressionForest::predict`). Sum is computed in `double` to avoid
   * loss-of-precision when many `float` outcomes are added. Throws on
   * empty input.
   */
  types::Outcome mean(std::vector<types::Outcome> const& preds);

  /**
   * @brief Map each label in @p groups to its index in iteration order.
   *
   * Used by `predict(FeatureMatrix, Proportions)` on `ClassificationTree`
   * and `ClassificationForest` to assign each group a column in the
   * proportions matrix. `std::set` iterates in ascending key order, so
   * the resulting indices are sorted by group label.
   */
  std::map<types::GroupId, int> group_indices(std::set<types::GroupId> const& groups);
}
