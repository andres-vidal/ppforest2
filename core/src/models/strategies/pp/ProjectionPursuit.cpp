#include "models/strategies/pp/ProjectionPursuit.hpp"

#include "models/strategies/NodeContext.hpp"
#include "utils/Invariant.hpp"

namespace ppforest2::pp {
  void ProjectionPursuit::optimize(NodeContext& ctx, stats::RNG& rng) const {
    if (ctx.aborted) {
      return;
    }

    compute(ctx, rng);

    if (ctx.aborted) {
      return;
    }

    invariant(ctx.projector.has_value(), "ProjectionPursuit must set ctx.projector");
    invariant(ctx.pp_index_value.has_value(), "ProjectionPursuit must set ctx.pp_index_value");

    if (ctx.projector.value().hasNaN()) {
      ctx.aborted = true;
    }
  }
}
