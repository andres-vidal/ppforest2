#include "models/strategies/binarize/Binarization.hpp"

#include "models/strategies/NodeContext.hpp"
#include "utils/Invariant.hpp"

namespace ppforest2::binarize {
  void Binarization::regroup(NodeContext& ctx, stats::RNG& rng) const {
    if (ctx.aborted) {
      return;
    }

    compute(ctx, rng);

    if (ctx.aborted) {
      return;
    }

    invariant(ctx.y_bin.has_value(), "Binarization must set ctx.y_bin");

    if (ctx.y_bin.value().groups.size() < 2) {
      ctx.aborted = true;
    }
  }
}
