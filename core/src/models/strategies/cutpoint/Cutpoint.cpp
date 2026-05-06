#include "models/strategies/cutpoint/Cutpoint.hpp"

#include "models/strategies/NodeContext.hpp"
#include "stats/GroupPartition.hpp"
#include "utils/Invariant.hpp"
#include "utils/Types.hpp"

#include <iterator>
#include <utility>

namespace ppforest2::cutpoint {

  namespace {
    /**
     * @brief Orient the two active-partition groups by projected mean, based on the cutpoint.
     *
     * Writes `ctx.lower_group` and `ctx.upper_group` so the former routes to the lower half-space and the later routes to the upper half-space. 
     */
    void orient_groups(NodeContext& ctx) {
      auto const& y_part   = ctx.active_partition();
      auto it              = y_part.groups.begin();
      types::GroupId lower = *it;
      types::GroupId upper = *std::next(it);

      types::Feature const mean_lower = y_part.group(ctx.x, lower).colwise().mean().dot(ctx.projector.value());
      types::Feature const mean_upper = y_part.group(ctx.x, upper).colwise().mean().dot(ctx.projector.value());

      if (mean_lower > mean_upper) {
        std::swap(lower, upper);
      }

      ctx.lower_group = lower;
      ctx.upper_group = upper;
    }
  }

  void Cutpoint::locate(NodeContext& ctx, stats::RNG& rng) const {
    if (ctx.aborted) {
      return;
    }

    compute(ctx, rng);

    if (ctx.aborted) {
      return;
    }

    invariant(ctx.cutpoint.has_value(), "Cutpoint must set ctx.cutpoint.");
    invariant(ctx.active_partition().groups.size() == 2, "Cutpoint expects a 2-group active partition.");
    invariant(ctx.projector.has_value(), "Cutpoint expects a projector to be set.");

    orient_groups(ctx);
  }
}
