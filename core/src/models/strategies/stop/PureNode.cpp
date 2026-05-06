#include "models/strategies/stop/PureNode.hpp"
#include "models/strategies/NodeContext.hpp"
#include "utils/JsonReader.hpp"

#include <nlohmann/json.hpp>

namespace ppforest2::stop {
  nlohmann::json PureNode::to_json() const {
    return {{"name", "pure_node"}};
  }

  bool PureNode::compute(NodeContext const& ctx, stats::RNG& /*rng*/) const {
    return ctx.y.groups.size() <= 1;
  }

  StopRule::Ptr pure_node() {
    return std::make_shared<PureNode>();
  }

  StopRule::Ptr PureNode::from_json(nlohmann::json const& j) {
    JsonReader{j, "pure_node"}.only_keys({"name"});
    return pure_node();
  }
}
