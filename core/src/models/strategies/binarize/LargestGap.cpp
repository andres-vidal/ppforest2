#include "models/strategies/binarize/LargestGap.hpp"

#include "models/strategies/NodeContext.hpp"
#include "utils/Invariant.hpp"
#include "utils/JsonReader.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <tuple>
#include <vector>

#include <nlohmann/json.hpp>

using namespace ppforest2::types;
using namespace ppforest2::stats;

namespace ppforest2::binarize {
  nlohmann::json LargestGap::to_json() const {
    return {{"name", "largest_gap"}};
  }

  void LargestGap::compute(NodeContext& ctx, RNG& /*rng*/) const {
    invariant(ctx.projector.has_value(), "LargestGap requires projector on NodeContext");
    FeatureMatrix const projected_x = ctx.x * ctx.projector.value();
    GroupPartition const& y         = ctx.y;

    std::vector<std::tuple<GroupId, Feature>> means;

    invariant(projected_x.cols() == 1, "Binary regrouping requires unidimensional data");

    for (GroupId const group : y.groups) {
      Feature const group_mean = y.group(projected_x, group).mean();
      means.emplace_back(group, group_mean);
    }

    std::stable_sort(means.begin(), means.end(), [](auto const& a, auto const& b) {
      return std::isless(std::get<1>(a), std::get<1>(b));
    });

    Feature edge_gap   = -1;
    GroupId edge_group = -1;

    for (size_t i = 0; i + 1 < means.size(); i++) {
      Feature const gap = std::get<1>(means[i + 1]) - std::get<1>(means[i]);

      if (gap > edge_gap) {
        edge_gap   = gap;
        edge_group = std::get<0>(means[i + 1]);
      }
    }

    if (edge_group == -1) {
      edge_group = std::get<0>(means.front());
    }

    std::map<GroupId, int> binary_mapping;

    bool edge_found = false;
    for (auto const& [group, mean] : means) {
      edge_found            = edge_found || group == edge_group;
      binary_mapping[group] = edge_found ? 1 : 0;
    }

    ctx.y_bin.emplace(y.remap(binary_mapping));
  }

  Binarization::Ptr largest_gap() {
    return std::make_shared<LargestGap>();
  }

  Binarization::Ptr LargestGap::from_json(nlohmann::json const& j) {
    JsonReader{j, "largest_gap"}.only_keys({"name"});
    return largest_gap();
  }
}
