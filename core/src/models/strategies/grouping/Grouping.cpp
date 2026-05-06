#include "models/strategies/grouping/Grouping.hpp"

#include "models/strategies/NodeContext.hpp"
#include "utils/Invariant.hpp"

namespace ppforest2::grouping {

  namespace {
    /**
     * @brief Whether the provided partition split represents no progress
     * 
     * Returns true if `ctx.lower_y_part` or `ctx.upper_y_part` contain the same amount or 
     * more observations than the current partition, which indicates the algorithm is unbounded.
     *
     */
    bool no_progress(NodeContext const& ctx) {
      int const parent_size = ctx.y.total_size();

      return ctx.lower_y_part.value().total_size() >= parent_size ||
             ctx.upper_y_part.value().total_size() >= parent_size;
    }
  }

  stats::GroupPartition Grouping::init(types::OutcomeVector const& y) const {
    invariant(y.size() > 0, "Grouping::init requires a non-empty response vector");

    stats::GroupPartition partition = compute_init(y);

    invariant(!partition.groups.empty(), "Grouping::init: partition must have at least one group");
    invariant(partition.group_start(partition.first_group()) == 0, "Grouping::init: partition must be rooted at row 0");
    invariant(partition.total_size() == y.size(), "Grouping::init: partition must cover every row of y");

    return partition;
  }

  void Grouping::split(NodeContext& ctx, stats::RNG& rng) const {
    if (ctx.aborted) {
      return;
    }

    invariant(ctx.lower_group.has_value(), "Grouping requires ctx.lower_group (set by Cutpoint)");
    invariant(ctx.upper_group.has_value(), "Grouping requires ctx.upper_group (set by Cutpoint)");

    compute(ctx, rng);

    invariant(ctx.lower_y_part.has_value(), "Grouping must set ctx.lower_y_part");
    invariant(ctx.upper_y_part.has_value(), "Grouping must set ctx.upper_y_part");

    ctx.aborted = no_progress(ctx);
  }
}
