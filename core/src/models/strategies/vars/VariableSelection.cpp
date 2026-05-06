#include "models/strategies/vars/VariableSelection.hpp"

#include "models/strategies/NodeContext.hpp"
#include "utils/Invariant.hpp"

namespace ppforest2::vars {
  void VariableSelection::select(NodeContext& ctx, stats::RNG& rng) const {
    if (ctx.aborted) {
      return;
    }

    compute(ctx, rng);

    if (ctx.aborted) {
      return;
    }

    invariant(ctx.var_selection.has_value(), "VariableSelection must set ctx.var_selection");
  }
}
