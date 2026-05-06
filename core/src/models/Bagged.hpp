#pragma once

#include "utils/Types.hpp"

#include <Eigen/Dense>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace ppforest2 {
  /**
   * @brief Bootstrap-aggregated model wrapper.
   *
   * Pairs a concrete model with the row indices of the bootstrap sample it
   * was trained on, so out-of-bag queries can recover the complementary
   * observations. Template over `M` so the bagging abstraction is
   * orthogonal to the model being aggregated — today `M = Tree`, but
   * nothing in this class is tree-specific, and a future `Bagged<GBTree>`
   * or `Bagged<LinearModel>` would reuse the exact same wrapper.
   *
   * `M` must provide:
   *   - `predict(types::FeatureVector const&)` returning a scalar outcome
   *   - `predict(types::FeatureMatrix const&)` returning an OutcomeVector
   *   - `bool degenerate` (field or method) for the forest retry logic
   *   - `operator==(M const&)` for structural equality round-trips
   *
   * Defined entirely in the header so consumers don't need an explicit
   * instantiation list — including this file with a complete `M` is
   * enough to use any `Bagged<M>`.
   */
  template<typename M> struct Bagged {
    using Ptr = std::unique_ptr<Bagged<M>>;

    /** @brief The bootstrap-trained model. */
    std::unique_ptr<M> model;
    /** @brief Row indices (into the original training set) used to train `model`. */
    std::vector<int> sample_indices;

    Bagged(std::unique_ptr<M> model, std::vector<int> sample_indices)
        : model(std::move(model))
        , sample_indices(std::move(sample_indices)) {}

    /**
     * @brief Construct a `Bagged<M>` and return it as a `Ptr`.
     *
     * Sugar for the `std::make_unique<Bagged<M>>(std::move(model),
     * std::move(sample_indices))` boilerplate at every callsite that
     * builds a bag (forest training, test fixtures, deserializers).
     * `Derived` defaults to `M`, but lets callers wrap a subclass
     * pointer (e.g. `ClassificationTree::Ptr`) without an upcast at
     * the callsite — the cast happens here in one place.
     */
    template<typename Derived = M> static Ptr make(std::unique_ptr<Derived> model, std::vector<int> sample_indices) {
      return std::make_unique<Bagged<M>>(std::move(model), std::move(sample_indices));
    }

    /** @brief Delegate single-row prediction to the wrapped model. */
    types::Outcome predict(types::FeatureVector const& x) const { return this->model->predict(x); }

    /** @brief Delegate batch prediction to the wrapped model. */
    types::OutcomeVector predict(types::FeatureMatrix const& x) const { return this->model->predict(x); }

    /**
     * @brief Row indices of training observations *not* in the bootstrap sample.
     *
     * O(n_sample + n_total) bitmap scan: a bootstrap bag has ~63% unique
     * entries, and `vector<uint8_t>` access is faster than the
     * `vector<bool>` proxy or a tree-set lookup. Out-of-range indices in
     * `sample_indices` are silently skipped — bootstrap sampling never
     * produces them.
     *
     * @param n_total  Total number of training observations.
     */
    std::vector<int> oob_indices(int n_total) const {
      if (n_total <= 0) {
        return {};
      }

      std::vector<std::uint8_t> in_bag(static_cast<std::size_t>(n_total), 0);

      for (int const i : sample_indices) {
        if (i >= 0 && i < n_total) {
          in_bag[static_cast<std::size_t>(i)] = 1;
        }
      }

      std::vector<int> oob;
      oob.reserve(static_cast<std::size_t>(n_total));
      for (int i = 0; i < n_total; ++i) {
        if (in_bag[static_cast<std::size_t>(i)] == 0) {
          oob.push_back(i);
        }
      }
      return oob;
    }

    /**
     * @brief Predict a subset of rows (typically OOB indices).
     *
     * The returned vector has the same size as @p row_idx; element `i`
     * is the wrapped model's prediction for row `row_idx[i]` of `x`.
     */
    types::OutcomeVector predict_oob(types::FeatureMatrix const& x, std::vector<int> const& row_idx) const {
      if (row_idx.empty()) {
        return types::OutcomeVector(0);
      }
      return model->predict(static_cast<types::FeatureMatrix>(x(row_idx, Eigen::all)));
    }

    /** @brief Whether the wrapped model reported a degenerate training run. */
    bool degenerate() const { return model->degenerate; }

    /**
     * @brief Structural equality on the wrapped model only.
     *
     * `sample_indices` is **deliberately excluded** from the comparison.
     * It records which rows were used to train this bag — bookkeeping for
     * OOB computation, not an identity property of the model. Two bags
     * that would produce the same predictions on every input are equal
     * here, even if they were trained on different bootstrap samples.
     *
     * Callers that need to assert sample-indices round-trip must compare
     * `sample_indices` directly (see `Json.test.cpp`'s round-trip test).
     */
    bool operator==(Bagged const& other) const { return *model == *other.model; }
    bool operator!=(Bagged const& other) const { return !(*this == other); }
  };

  /**
   * @brief Apply @p fn to every bag in @p bags that has usable OOB rows.
   *
   * Skips bags whose bootstrap sample happened to cover every training row
   * (no OOB to evaluate on). Bags whose `degenerate()` is true are *not*
   * filtered: that flag propagates from any descendant leaf, so a tree with
   * one deep aborted split still has valid upper branches and meaningful
   * `predict()`. OOB-based callers (VI permuted/weighted) downweight or
   * zero-out fully-degenerate trees naturally via their score function
   * (e.g. NMSE clipping for regression).
   *
   * The callback receives `(bag, oob_idx, k)` where `k` is the bag's
   * position in the container — needed by callers that derive a per-bag
   * RNG seed from `seed ^ k`. Iteration order matches the container's
   * order. Use `return;` inside @p fn to skip the rest of the body for
   * the current bag (the lambda's `return` acts as `continue`).
   */
  template<typename M, typename F>
  void for_each_bag_with_oob(std::vector<std::unique_ptr<Bagged<M>>> const& bags, int n_total, F fn) {
    int const n = static_cast<int>(bags.size());
    for (int k = 0; k < n; ++k) {
      Bagged<M> const& bag     = *bags[static_cast<std::size_t>(k)];
      std::vector<int> oob_idx = bag.oob_indices(n_total);
      if (oob_idx.empty()) {
        continue;
      }
      fn(bag, oob_idx, k);
    }
  }
}
