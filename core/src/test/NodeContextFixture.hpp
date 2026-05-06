#pragma once

#include "models/strategies/NodeContext.hpp"
#include "stats/GroupPartition.hpp"
#include "stats/Stats.hpp"
#include "utils/Types.hpp"

#include <utility>

namespace ppforest2::test {
  namespace {
    /**
     * @brief Build a median-split `GroupIdVector` of size @p n.
     *
     * Mirrors what `ByCutpoint::init` produces for a regression root:
     * the first `n/2` entries go to group 0, the rest to group 1. Used
     * by the regression-flavoured `NodeContextFixture` constructor so
     * tests don't need to redeclare the standard 2-group partition that
     * regression's initial state always carries.
     */
    inline types::GroupIdVector median_split_ids(int n) {
      types::GroupIdVector ids(n);
      int const mid = n / 2;
      for (int i = 0; i < n; ++i) {
        ids(i) = (i < mid) ? 0 : 1;
      }
      return ids;
    }
  }

  /**
   * @brief Test-only fixture that owns the storage backing a `NodeContext`.
   *
   * `NodeContext` holds non-owning references to a feature matrix, group
   * partition, and outcome vector, so a test must keep those alive
   * separately. This fixture bundles them with an RNG and a fully-wired
   * `NodeContext`, so each strategy NVI test can stop redeclaring the
   * boilerplate.
   *
   * Three constructors:
   *
   * - `(x, GroupIdVector ids)` — classification flavour. `y_vec` is
   *   derived from `ids` via `cast<Outcome>()`; the partition is
   *   `GroupPartition(ids)`.
   * - `(x, OutcomeVector y_vec)` — regression flavour. `y_ids` and the
   *   partition are a median split matching `ByCutpoint::init`'s
   *   behaviour at the root, so tests pass only the continuous response
   *   they actually care about.
   * - `(x, GroupPartition, OutcomeVector y_vec)` — escape hatch for
   *   tests that need a partition shape neither default produces (e.g.
   *   `gp.subset(...)`, custom `two_groups(...)` bounds). `y_ids` is
   *   left empty in this case since the partition came in pre-built.
   *
   * Public fields so tests can mutate them after construction
   * (e.g. `f.x = …`, `f.ctx.projector = …`).
   *
   * Non-copyable / non-movable: `ctx` references the other members of
   * the same fixture, so any copy or move would silently produce
   * dangling references.
   */
  struct NodeContextFixture {
    types::FeatureMatrix x;
    types::GroupIdVector y_ids;
    stats::GroupPartition y_part;
    types::OutcomeVector y_vec;
    stats::RNG rng;
    NodeContext ctx;

    NodeContextFixture(types::FeatureMatrix x_in, types::GroupIdVector ids_in)
        : x(std::move(x_in))
        , y_ids(std::move(ids_in))
        , y_part(y_ids)
        , y_vec(y_ids.cast<types::Outcome>())
        , rng(0)
        , ctx(x, y_part, y_vec, 0) {}

    NodeContextFixture(types::FeatureMatrix x_in, types::OutcomeVector y_vec_in)
        : x(std::move(x_in))
        , y_ids(median_split_ids(static_cast<int>(y_vec_in.size())))
        , y_part(y_ids)
        , y_vec(std::move(y_vec_in))
        , rng(0)
        , ctx(x, y_part, y_vec, 0) {}

    NodeContextFixture(types::FeatureMatrix x_in, stats::GroupPartition y_part_in, types::OutcomeVector y_vec_in)
        : x(std::move(x_in))
        , y_ids()
        , y_part(std::move(y_part_in))
        , y_vec(std::move(y_vec_in))
        , rng(0)
        , ctx(x, y_part, y_vec, 0) {}

    NodeContextFixture(NodeContextFixture const&)            = delete;
    NodeContextFixture(NodeContextFixture&&)                 = delete;
    NodeContextFixture& operator=(NodeContextFixture const&) = delete;
    NodeContextFixture& operator=(NodeContextFixture&&)      = delete;
  };
}
