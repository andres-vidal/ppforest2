#include "models/strategies/stop/MaxDepth.hpp"
#include "utils/JsonReader.hpp"

#include "models/strategies/NodeContext.hpp"
#include "utils/UserError.hpp"

#include <nlohmann/json.hpp>
#include <string>

namespace ppforest2::stop {
  MaxDepth::MaxDepth(int max_depth)
      : max_depth(max_depth) {
    user_error(max_depth >= 0, "stop_max_depth: max_depth must be >= 0");
  }

  nlohmann::json MaxDepth::to_json() const {
    return {{"name", "max_depth"}, {"max_depth", max_depth}};
  }

  std::string MaxDepth::display_name() const {
    return "Max depth (" + std::to_string(max_depth) + ")";
  }

  bool MaxDepth::compute(NodeContext const& ctx, stats::RNG& /*rng*/) const {
    return ctx.depth >= max_depth;
  }

  StopRule::Ptr max_depth(int n) {
    return std::make_shared<MaxDepth>(n);
  }

  StopRule::Ptr MaxDepth::from_json(nlohmann::json const& j) {
    JsonReader const r{j, "max_depth"};
    r.only_keys({"name", "max_depth"});
    return stop::max_depth(static_cast<int>(r.require_int("max_depth", 0)));
  }
}
