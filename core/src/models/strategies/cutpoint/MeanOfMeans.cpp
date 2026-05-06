#include "models/strategies/cutpoint/MeanOfMeans.hpp"

#include "models/strategies/NodeContext.hpp"
#include "utils/Invariant.hpp"
#include "utils/JsonReader.hpp"

#include <nlohmann/json.hpp>

using namespace ppforest2::types;
using namespace ppforest2::stats;

namespace ppforest2::cutpoint {
  nlohmann::json MeanOfMeans::to_json() const {
    return {{"name", "mean_of_means"}};
  }

  void MeanOfMeans::compute(NodeContext& ctx, RNG& /*rng*/) const {
    invariant(ctx.projector.has_value(), "MeanOfMeans requires projector on NodeContext");
    auto const& y_part    = ctx.active_partition();
    auto g1               = *y_part.groups.begin();
    auto g2               = *std::next(y_part.groups.begin());
    auto data_1           = y_part.group(ctx.x, g1);
    auto data_2           = y_part.group(ctx.x, g2);
    auto const& projector = ctx.projector.value();
    ctx.cutpoint          = ((data_1 * projector).mean() + (data_2 * projector).mean()) / 2;
  }

  Cutpoint::Ptr mean_of_means() {
    return std::make_shared<MeanOfMeans>();
  }

  Cutpoint::Ptr MeanOfMeans::from_json(nlohmann::json const& j) {
    JsonReader{j, "mean_of_means"}.only_keys({"name"});
    return mean_of_means();
  }
}
