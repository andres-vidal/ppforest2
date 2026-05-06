#include "models/strategies/stop/StopRule.hpp"

#include "models/strategies/NodeContext.hpp"
#include "stats/GroupPartition.hpp"

namespace ppforest2::stop {
  bool StopRule::should_stop(NodeContext const& ctx, stats::RNG& rng) const {
    if (ctx.y.groups.size() < 2) {
      return true;
    }

    return compute(ctx, rng);
  }
}
