#pragma once

#include "models/Projector.hpp"
#include "models/strategies/vars/VariableSelection.hpp"
#include "stats/GroupPartition.hpp"
#include "utils/Types.hpp"

#include <optional>

using Projector = ppforest2::pp::Projector;

namespace ppforest2 {
  /**
   * @brief Mutable context accumulating intermediate results during node training.
   *
   * Starts with node-level data (x, y, depth) and accumulates results
   * as each strategy in the training pipeline executes. Each strategy
   * reads what it needs and writes its results back.
   */
  struct NodeContext {
    /**
     * @brief Full feature matrix.
     *
     * Non-const because regression's `ByCutpoint` grouping strategy
     * reorders rows in place within each node's range. Classification
     * strategies only read — there's no mode-specific subtype, so the
     * const promise lives at the strategy level, not in the context.
     */
    types::FeatureMatrix& x;
    /** @brief Original G-group partition for this node. */
    stats::GroupPartition const& y;
    /** @brief Depth of this node in the tree. */
    int depth;

    /**
     * @brief Sticky abort flag for the per-node strategy pipeline.
     *
     * When a producer step detects a degenerate outcome (NaN projector,
     * fewer than 2 binary groups, etc.), it sets this flag. Subsequent
     * strategies called on the same context skip their work — every
     * strategy's public NVI entry point checks this first before
     * dispatching to the subclass `compute`. The orchestrator
     * (`Tree::build_root`) then converts the aborted context into a
     * degenerate leaf with a single check at the end of the node step.
     */
    bool aborted = false;

    /**
     * @brief Raw response vector whose row order matches `x`.
     *
     * Regression strategies (`MeanResponse` leaf, `MinVariance` stop,
     * `ByCutpoint` grouping) read or reorder it; classification
     * strategies ignore it. Non-const for the same reason as `x`.
     */
    types::OutcomeVector& y_vec;

    /** @brief Set by select_vars: variable selection result. `std::nullopt` before select_vars runs. */
    std::optional<vars::VariableSelection::Result> var_selection;

    /** @brief Set by find_projection: optimized projector (full dimension, expanded). `std::nullopt` before find_projection runs. */
    std::optional<Projector> projector;
    /** @brief Set by find_projection: projection pursuit index value achieved. `std::nullopt` before find_projection runs. */
    std::optional<types::Feature> pp_index_value;

    /**
     * @brief Set by regroup (multiclass → binary): 2-group binarized partition.
     *
     * `std::nullopt` on binary nodes (no binarization needed). Consumers
     * should read via `active_partition()`, which falls back to `y`.
     */
    std::optional<stats::GroupPartition> y_bin;

    /** @brief Set by find_cutpoint: split cutpoint in projected space. `std::nullopt` before find_cutpoint runs. */
    std::optional<types::Feature> cutpoint;

    /**
     * @brief Set by find_cutpoint: labels of the two groups in `active_partition()`,
     *        oriented so `lower_group`'s projected mean < `upper_group`'s.
     *
     * Consumed by `Grouping::split` to route rows to lower/upper children.
     */
    std::optional<types::GroupId> lower_group;
    std::optional<types::GroupId> upper_group;

    /** @brief Set by group: child partitions routed to the lower / upper child nodes. */
    std::optional<stats::GroupPartition> lower_y_part;
    std::optional<stats::GroupPartition> upper_y_part;

    NodeContext(types::FeatureMatrix& x, stats::GroupPartition const& y, types::OutcomeVector& y_vec, int depth)
        : x(x)
        , y(y)
        , depth(depth)
        , y_vec(y_vec) {}

    /**
     * @brief Return the active group partition.
     *
     * After binarization, returns the binary partition (2 groups).
     * Before binarization (or for 2-group nodes), returns the original partition.
     */
    stats::GroupPartition const& active_partition() const { return y_bin.has_value() ? *y_bin : y; }
  };
}
