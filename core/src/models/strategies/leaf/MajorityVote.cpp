#include "models/strategies/leaf/MajorityVote.hpp"
#include "models/TreeLeaf.hpp"
#include "models/strategies/NodeContext.hpp"

#include "utils/JsonReader.hpp"

#include <nlohmann/json.hpp>

using namespace ppforest2::types;
using namespace ppforest2::stats;

namespace ppforest2::leaf {
  nlohmann::json MajorityVote::to_json() const {
    return {{"name", "majority_vote"}};
  }

  TreeNode::Ptr MajorityVote::compute(NodeContext const& ctx, RNG& /*rng*/) const {
    GroupId majority  = ctx.y.first_group();
    int majority_size = 0;

    for (auto const& g : ctx.y.groups) {
      int const sz = ctx.y.group_size(g);

      if (sz > majority_size) {
        majority_size = sz;
        majority      = g;
      }
    }

    return TreeLeaf::make(majority);
  }

  LeafStrategy::Ptr majority_vote() {
    return std::make_shared<MajorityVote>();
  }

  LeafStrategy::Ptr MajorityVote::from_json(nlohmann::json const& j) {
    JsonReader{j, "majority_vote"}.only_keys({"name"});
    return majority_vote();
  }
}
