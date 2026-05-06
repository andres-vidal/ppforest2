#include "models/strategies/vars/Uniform.hpp"

#include "models/strategies/NodeContext.hpp"
#include "stats/Uniform.hpp"
#include "utils/RangeVector.hpp"
#include "utils/Invariant.hpp"
#include "utils/JsonReader.hpp"

#include <limits>

namespace ppforest2::vars {
  Uniform::Uniform(int n_vars)
      : n_vars(n_vars) {
    invariant(n_vars > 0, "The number of variables must be greater than 0.");
  }

  nlohmann::json Uniform::to_json() const {
    return {{"name", "uniform"}, {"count", n_vars}};
  }

  void Uniform::compute(NodeContext& ctx, stats::RNG& rng) const {
    invariant(
        n_vars <= ctx.x.cols(),
        "The number of variables must be less than or equal to the number of columns in the data."
    );

    if (n_vars == ctx.x.cols()) {
      ctx.var_selection = VariableSelection::Result(utils::range_vector(ctx.x.cols()), ctx.x.cols());
      return;
    }

    stats::Uniform unif(0, static_cast<int>(ctx.x.cols()) - 1);
    ctx.var_selection = VariableSelection::Result(unif.distinct(n_vars, rng), ctx.x.cols());
  }

  VariableSelection::Ptr uniform(int n_vars) {
    return std::make_shared<Uniform>(n_vars);
  }

  VariableSelection::Ptr Uniform::from_json(nlohmann::json const& j) {
    JsonReader const r{j, "uniform"};
    r.only_keys({"name", "count", "proportion"});

    if (r.contains("proportion")) {
      // `require_number` enforces the (0, 1] range (we use [tiny, 1.0] to
      // express an open lower bound — exactly zero is rejected).
      (void)r.require_number("proportion", std::numeric_limits<double>::min(), 1.0);
      // Proportion is resolved to count later when total_vars is known.
      // Return a placeholder — caller must resolve before use.
      return uniform(1);
    }

    return uniform(static_cast<int>(r.require_int("count", 1)));
  }
}
