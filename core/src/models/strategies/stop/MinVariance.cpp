#include "models/strategies/stop/MinVariance.hpp"

#include "models/strategies/NodeContext.hpp"
#include "stats/Stats.hpp"
#include "utils/Invariant.hpp"
#include "utils/JsonReader.hpp"

#include <Eigen/Dense>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <sstream>

using namespace ppforest2::types;

namespace ppforest2::stop {
  MinVariance::MinVariance(Feature threshold)
      : threshold(threshold) {}

  nlohmann::json MinVariance::to_json() const {
    return {{"name", "min_variance"}, {"threshold", threshold}};
  }

  std::string MinVariance::display_name() const {
    std::ostringstream oss;
    oss << "Min variance (" << std::defaultfloat << std::setprecision(6) << threshold << ")";
    return oss.str();
  }

  bool MinVariance::compute(NodeContext const& ctx, stats::RNG& /*rng*/) const {

    Eigen::VectorXd const y = ctx.y.data(ctx.y_vec).template cast<double>();

    if (y.size() <= 1) {
      return true;
    }

    return stats::var(y) < static_cast<double>(threshold);
  }

  StopRule::Ptr min_variance(Feature threshold) {
    return std::make_shared<MinVariance>(threshold);
  }

  StopRule::Ptr MinVariance::from_json(nlohmann::json const& j) {
    JsonReader const r{j, "min_variance"};
    r.only_keys({"name", "threshold"});
    return min_variance(static_cast<Feature>(r.require_number("threshold", 0.0)));
  }
}
