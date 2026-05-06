#include "models/strategies/grouping/ByLabel.hpp"
#include "utils/JsonReader.hpp"

#include "models/strategies/NodeContext.hpp"

#include <nlohmann/json.hpp>

using namespace ppforest2::types;
using namespace ppforest2::stats;

namespace ppforest2::grouping {
  nlohmann::json ByLabel::to_json() const {
    return {{"name", "by_label"}};
  }

  GroupPartition ByLabel::compute_init(OutcomeVector const& y) const {
    return GroupPartition(y);
  }

  void ByLabel::compute(NodeContext& ctx, RNG& /*rng*/) const {
    auto const lower = ctx.lower_group.value();
    auto const upper = ctx.upper_group.value();

    auto const& y_part = ctx.active_partition();
    ctx.lower_y_part.emplace(y_part.subset(y_part.subgroups.at(lower)));
    ctx.upper_y_part.emplace(y_part.subset(y_part.subgroups.at(upper)));
  }

  Grouping::Ptr by_label() {
    return std::make_shared<ByLabel>();
  }

  Grouping::Ptr ByLabel::from_json(nlohmann::json const& j) {
    JsonReader{j, "by_label"}.only_keys({"name"});
    return by_label();
  }
}
