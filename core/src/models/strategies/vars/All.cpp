#include "models/strategies/vars/All.hpp"
#include "models/strategies/NodeContext.hpp"

#include "utils/RangeVector.hpp"
#include "utils/JsonReader.hpp"

#include <nlohmann/json.hpp>

namespace ppforest2::vars {
  nlohmann::json All::to_json() const {
    return {{"name", "all"}};
  }

  void All::compute(NodeContext& ctx, stats::RNG& /*rng*/) const {
    ctx.var_selection = VariableSelection::Result(utils::range_vector(ctx.x.cols()), ctx.x.cols());
  }

  VariableSelection::Ptr all() {
    return std::make_shared<All>();
  }

  VariableSelection::Ptr All::from_json(nlohmann::json const& j) {
    JsonReader{j, "all"}.only_keys({"name"});
    return all();
  }
}
