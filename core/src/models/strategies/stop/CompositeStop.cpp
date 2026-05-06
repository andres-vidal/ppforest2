#include "models/strategies/stop/CompositeStop.hpp"
#include "utils/JsonReader.hpp"

#include "models/strategies/NodeContext.hpp"
#include "utils/UserError.hpp"

#include <algorithm>
#include <iterator>
#include <nlohmann/json.hpp>

namespace ppforest2::stop {
  CompositeStop::CompositeStop(std::vector<StopRule::Ptr> rules)
      : rules(std::move(rules)) {
    user_error(!this->rules.empty(), "stop_any: rules must be non-empty. An empty composite would never stop.");
  }

  nlohmann::json CompositeStop::to_json() const {
    nlohmann::json j;
    j["name"] = "any";

    nlohmann::json rule_array = nlohmann::json::array();

    for (auto const& rule : rules) {
      rule_array.push_back(rule->to_json());
    }

    j["rules"] = rule_array;
    return j;
  }

  std::string CompositeStop::display_name() const {
    std::string result = "Any(";

    for (std::size_t i = 0; i < rules.size(); ++i) {
      if (i > 0) {
        result += ", ";
      }

      result += rules[i]->display_name();
    }

    result += ")";

    return result;
  }

  std::set<types::Mode> CompositeStop::supported_modes() const {
    // Start with both modes; intersect with each child's supported modes.
    std::set<types::Mode> modes = {types::Mode::Classification, types::Mode::Regression};

    for (auto const& rule : rules) {
      std::set<types::Mode> child = rule->supported_modes();
      std::set<types::Mode> inter;
      std::set_intersection(
          modes.begin(), modes.end(), child.begin(), child.end(), std::inserter(inter, inter.begin())
      );
      modes = std::move(inter);
    }

    return modes;
  }

  bool CompositeStop::compute(NodeContext const& ctx, stats::RNG& rng) const {
    return std::any_of(rules.begin(), rules.end(), [&](auto const& rule) { return rule->should_stop(ctx, rng); });
  }

  StopRule::Ptr any(std::vector<StopRule::Ptr> rules) {
    return std::make_shared<CompositeStop>(std::move(rules));
  }

  StopRule::Ptr CompositeStop::from_json(nlohmann::json const& j) {
    JsonReader const r{j, "any"};
    r.only_keys({"name", "rules"});

    std::vector<StopRule::Ptr> rules;

    for (auto const& rule_json : r.require_array("rules")) {
      rules.push_back(StopRule::from_json(rule_json));
    }

    return any(std::move(rules));
  }
}
