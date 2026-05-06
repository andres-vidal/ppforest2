#include "models/strategies/stop/MinSize.hpp"
#include "utils/JsonReader.hpp"

#include "models/strategies/NodeContext.hpp"
#include "utils/UserError.hpp"

#include <nlohmann/json.hpp>
#include <string>

namespace ppforest2::stop {
  MinSize::MinSize(int min_size)
      : min_size(min_size) {
    user_error(min_size >= 2, "stop_min_size: min_size must be >= 2");
  }

  nlohmann::json MinSize::to_json() const {
    return {{"name", "min_size"}, {"min_size", min_size}};
  }

  std::string MinSize::display_name() const {
    return "Min size (" + std::to_string(min_size) + ")";
  }

  bool MinSize::compute(NodeContext const& ctx, stats::RNG& /*rng*/) const {
    int total = 0;

    for (auto const& g : ctx.y.groups) {
      total += ctx.y.group_size(g);
    }

    return total < min_size;
  }

  StopRule::Ptr min_size(int n) {
    return std::make_shared<MinSize>(n);
  }

  StopRule::Ptr MinSize::from_json(nlohmann::json const& j) {
    JsonReader const r{j, "min_size"};
    r.only_keys({"name", "min_size"});
    return stop::min_size(static_cast<int>(r.require_int("min_size", 2)));
  }
}
