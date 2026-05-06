#include "models/strategies/binarize/Disabled.hpp"

#include "models/strategies/NodeContext.hpp"
#include "utils/JsonReader.hpp"

#include <nlohmann/json.hpp>

namespace ppforest2::binarize {
  nlohmann::json Disabled::to_json() const {
    return {{"name", "disabled"}};
  }

  void Disabled::compute(NodeContext& /*ctx*/, stats::RNG& /*rng*/) const {
    throw std::runtime_error("binarize::Disabled was invoked. This placeholder binarizer "
                             "should never fire: either the grouping strategy produced a "
                             ">2-group partition (use binarize::largest_gap or another real binarizer instead), "
                             "or the spec was assembled incorrectly.");
  }

  Binarization::Ptr disabled() {
    return std::make_shared<Disabled>();
  }

  Binarization::Ptr Disabled::from_json(nlohmann::json const& j) {
    JsonReader const r{j, "disabled"};
    r.only_keys({"name"});
    return disabled();
  }
}
