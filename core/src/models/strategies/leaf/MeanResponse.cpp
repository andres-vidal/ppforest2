#include "models/strategies/leaf/MeanResponse.hpp"

#include "models/TreeLeaf.hpp"
#include "models/strategies/NodeContext.hpp"
#include "utils/Invariant.hpp"
#include "utils/JsonReader.hpp"

#include <Eigen/Dense>
#include <nlohmann/json.hpp>

using namespace ppforest2::types;
using namespace ppforest2::stats;

namespace ppforest2::leaf {
  nlohmann::json MeanResponse::to_json() const {
    return {{"name", "mean_response"}};
  }

  TreeNode::Ptr MeanResponse::compute(NodeContext const& ctx, RNG& /*rng*/) const {

    Eigen::VectorXd const y = ctx.y.data(ctx.y_vec).template cast<double>();

    return TreeLeaf::make(static_cast<Feature>(y.mean()));
  }

  LeafStrategy::Ptr mean_response() {
    return std::make_shared<MeanResponse>();
  }

  LeafStrategy::Ptr MeanResponse::from_json(nlohmann::json const& j) {
    JsonReader{j, "mean_response"}.only_keys({"name"});
    return mean_response();
  }
}
