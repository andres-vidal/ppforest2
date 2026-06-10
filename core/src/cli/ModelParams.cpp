/**
 * @file ModelParams.cpp
 * @brief Model parameter construction, resolution, and defaults.
 */
#include "cli/ModelParams.hpp"
#include "cli/Validation.hpp"
#include "models/TrainingSpec.hpp"
#include "models/strategies/Strategy.hpp"
#include "serialization/JsonOptional.hpp"
#include "utils/Invariant.hpp"
#include "utils/Math.hpp"
#include "utils/UserError.hpp"

#include <charconv>
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <regex>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ppforest2::cli {

  namespace {
    void apply_strategy(
        nlohmann::json const& obj, std::string const& key, nlohmann::json& target_config, std::string& target_input
    ) {
      if (!obj.contains(key)) {
        return;
      }

      if (obj[key].is_object()) {
        target_config = obj[key];
      } else if (obj[key].is_string()) {
        target_input = obj[key].get<std::string>();
      }
    }

    nlohmann::json parse_scalar(std::string const& s) {
      int ival;
      auto [iptr, iec] = std::from_chars(s.data(), s.data() + s.size(), ival);

      if (iec == std::errc{} && iptr == s.data() + s.size()) {
        return ival;
      }

      char* end   = nullptr;
      double dval = std::strtod(s.c_str(), &end);

      if (end == s.c_str() + s.size()) {
        return dval;
      }

      return s;
    }

    void try_parse_strategy(std::string const& input, std::string const& flag, nlohmann::json& config) {
      if (input.empty()) {
        return;
      }

      try {
        config = strategy_string_to_json(input);
      } catch (std::exception const& e) {
        throw ppforest2::UserError(fmt::format("Invalid --{} value: {}", flag, e.what()));
      }
    }

    // Single bare token (no `=`, no `,`) maps to the strategy's primary
    // param — e.g. `min_size:5` → `{name: min_size, min_size: 5}`.
    // Numeric-only so `pda:lambda` stays a "missing =" error rather than
    // silently producing `{lambda: "lambda"}`.
    bool try_apply_positional_shorthand(std::string const& name, std::string const& token, nlohmann::json& j) {
      if (token.find('=') != std::string::npos || token.find(',') != std::string::npos) {
        return false;
      }

      auto const param = ppforest2::strategies::primary_param_for(name);
      if (!param) {
        return false;
      }

      nlohmann::json const val = parse_scalar(token);
      if (!val.is_number()) {
        return false;
      }

      j[*param] = val;
      return true;
    }

    void apply_kv_token(std::string const& token, nlohmann::json& j) {
      static std::regex const kv_pattern(R"(([^=]+)=(.*))");
      std::smatch kv;
      if (!std::regex_match(token, kv, kv_pattern)) {
        throw std::runtime_error("Invalid parameter (expected key=value): " + token);
      }
      j[kv[1].str()] = parse_scalar(kv[2].str());
    }

    // Wrap repeated `--stop` rules in a CompositeStop `any`-rule. Matches
    // the JSON shape produced by R's `stop_any(...)` and `CompositeStop::to_json()`.
    nlohmann::json wrap_stop_rules_as_any(std::vector<std::string> const& inputs) {
      nlohmann::json any_rule = {{"name", "any"}, {"rules", nlohmann::json::array()}};
      for (auto const& s : inputs) {
        try {
          any_rule["rules"].push_back(strategy_string_to_json(s));
        } catch (std::exception const& e) {
          throw ppforest2::UserError(fmt::format("Invalid --stop value '{}': {}", s, e.what()));
        }
      }
      return any_rule;
    }

  }

  float parse_proportion(std::string const& input) {
    try {
      auto slash_pos = input.find('/');

      if (slash_pos != std::string::npos) {
        int numerator   = std::stoi(input.substr(0, slash_pos));
        int denominator = std::stoi(input.substr(slash_pos + 1));

        user_error(denominator > 0, "fraction denominator must be positive");
        user_error(numerator > 0, "fraction numerator must be positive");

        float val = static_cast<float>(numerator) / static_cast<float>(denominator);

        user_error(val <= 1, "fraction must evaluate to a proportion in (0, 1]");

        return val;
      }

      float val = std::stof(input);

      user_error(val > 0 && val <= 1, "proportion must be in (0, 1]");

      return val;
    } catch (ppforest2::UserError const&) {
      throw;
    } catch (std::exception const& e) {
      throw ppforest2::UserError(fmt::format("Invalid proportion '{}': {}", input, e.what()));
    }
  }

  float parse_proportion(nlohmann::json const& j) {
    if (j.is_string()) {
      return parse_proportion(j.get<std::string>());
    }

    float val = j.get<float>();

    user_error(val > 0 && val <= 1, "proportion must be in (0, 1]");

    return val;
  }

  nlohmann::json strategy_string_to_json(std::string const& input) {
    static std::regex const pattern(R"(([^:]+)(?::(.+))?)");

    std::smatch match;

    if (!std::regex_match(input, match, pattern)) {
      throw std::runtime_error("Invalid strategy string: " + input);
    }

    nlohmann::json j;
    std::string const name = match[1].str();
    j["name"]              = name;

    if (!match[2].matched) {
      return j;
    }

    std::string const params_str = match[2].str();

    if (try_apply_positional_shorthand(name, params_str, j)) {
      return j;
    }

    static std::regex const token_pattern(R"([^,]+)");
    auto begin = std::sregex_iterator(params_str.begin(), params_str.end(), token_pattern);
    auto end   = std::sregex_iterator();

    for (auto it = begin; it != end; ++it) {
      apply_kv_token((*it).str(), j);
    }

    return j;
  }

  void ModelParams::validate(nlohmann::json const& config, std::vector<std::string>& errors) {
    // Mode
    if (config.contains("mode")) {
      if (config["mode"].is_string()) {
        auto mode = config["mode"].get<std::string>();
        check(
            mode == "classification" || mode == "regression", "mode must be 'classification' or 'regression'", errors
        );
      } else {
        errors.emplace_back("mode must be a string");
      }
    }

    // Size
    if (config.contains("size")) {
      check(config["size"].is_number_integer(), "size must be an integer", errors);

      if (config["size"].is_number_integer()) {
        check(config["size"].get<int>() >= 0, "size must be >= 0", errors);
      }
    } else {
      errors.push_back("size is required");
    }

    // Lambda
    if (config.contains("lambda") && config["lambda"].is_number()) {
      auto lambda = config["lambda"].get<float>();
      check(lambda >= 0 && lambda <= 1, "lambda must be in [0, 1]", errors);
    }

    // PP strategy vs lambda: at least one must be present
    check(config.contains("pp") || config.contains("lambda"), "lambda is required (or provide a pp strategy)", errors);

    // Variable selection
    if (config.contains("p_vars") && config["p_vars"].is_number()) {
      auto pv = config["p_vars"].get<float>();
      check(pv > 0 && pv <= 1, "p_vars must be in (0, 1]", errors);
    }

    bool has_vars_strategy = config.contains("vars") && !config["vars"].is_null();

    if (config.contains("n_vars") && config["n_vars"].is_number_integer() && !has_vars_strategy) {
      check(config["n_vars"].get<int>() > 0, "n_vars must be positive", errors);
    }

    // Threads
    if (config.contains("threads") && config["threads"].is_number_integer()) {
      check(config["threads"].get<int>() > 0, "threads must be positive", errors);
    }

    // Max retries
    if (config.contains("max_retries") && config["max_retries"].is_number_integer()) {
      check(config["max_retries"].get<int>() >= 0, "max_retries must be non-negative", errors);
    }
  }

  ModelParams::ModelParams(nlohmann::json const& config) {
    mode_input  = config.value("mode", mode_input);
    size        = config.value("size", size);
    lambda      = config.value("lambda", lambda);
    seed        = config.value("seed", seed);
    threads     = config.value("threads", threads);
    max_retries = config.value("max_retries", max_retries);
    n_vars      = config.value("n_vars", n_vars);

    if (config.contains("p_vars")) {
      auto const& v = config["p_vars"];
      p_vars_input  = v.is_string() ? v.get<std::string>() : v.dump();
    }

    apply_strategy(config, "pp", pp_config, pp_input);
    apply_strategy(config, "vars", vars_config, vars_input);
    apply_strategy(config, "cutpoint", cutpoint_config, cutpoint_input);
    // `stop` is object-only from a JSON config (CLI --stop's repeat
    // semantics live in `resolve()`).
    if (config.contains("stop") && config["stop"].is_object()) {
      stop_config = config["stop"];
    }
    apply_strategy(config, "binarize", binarize_config, binarize_input);
    apply_strategy(config, "grouping", grouping_config, grouping_input);
    apply_strategy(config, "leaf", leaf_config, leaf_input);
  }

  void ModelParams::resolve() {
    try_parse_strategy(pp_input, "pp", pp_config);
    try_parse_strategy(vars_input, "vars", vars_config);
    try_parse_strategy(cutpoint_input, "cutpoint", cutpoint_config);
    // Single --stop: parse as-is. Multiple: wrap in a CompositeStop `any` rule.
    if (stop_inputs.size() == 1) {
      try_parse_strategy(stop_inputs.front(), "stop", stop_config);
    } else if (stop_inputs.size() > 1) {
      stop_config = wrap_stop_rules_as_any(stop_inputs);
    }
    try_parse_strategy(binarize_input, "binarize", binarize_config);
    try_parse_strategy(grouping_input, "grouping", grouping_config);
    try_parse_strategy(leaf_input, "leaf", leaf_config);

    if (!p_vars_input.empty()) {
      p_vars = parse_proportion(p_vars_input);
    }
  }

  void ModelParams::resolve_defaults(unsigned int total_vars) {
    // clang-format off
    if (!threads) {
      #ifdef _OPENMP
      threads = omp_get_max_threads();
      #else
      threads = 1;
      #endif
    }
    // clang-format on

    if (total_vars == 0) {
      return;
    }

    // Resolve implicit APIs into explicit strategy configs
    if (pp_config.is_null()) {
      pp_config = {{"name", "pda"}, {"lambda", lambda}};
    }

    if (vars_config.is_null() && size > 0) {
      if (n_vars) {
        p_vars      = static_cast<float>(*n_vars) / static_cast<float>(total_vars);
        vars_config = {{"name", "uniform"}, {"count", *n_vars}};
      } else if (p_vars) {
        n_vars      = math::proportion_to_count(*p_vars, total_vars);
        vars_config = {{"name", "uniform"}, {"count", *n_vars}};
      } else {
        p_vars      = 0.5F;
        n_vars      = math::proportion_to_count(*p_vars, total_vars);
        vars_config = {{"name", "uniform"}, {"count", *n_vars}};
      }
    }

    // Resolve proportion to count in explicit vars config
    if (!vars_config.is_null() && vars_config.contains("proportion")) {
      int count = math::proportion_to_count(vars_config["proportion"].get<float>(), total_vars);
      vars_config.erase("proportion");
      vars_config["count"] = count;
    }

    invariant(
        !mode_input.empty(),
        "ModelParams::resolve_defaults: mode_input must be set before resolving strategy defaults "
        "(set via --mode, the config file, or Params::read_data)."
    );

    auto builder = TrainingSpec::builder(types::mode_from_string(mode_input));
    builder.apply_defaults();

    auto fill = [](nlohmann::json& config, auto const& strategy) {
      if (config.is_null()) {
        config = strategy->to_json();
      }
    };

    fill(pp_config, builder.config.pp);
    fill(vars_config, builder.config.vars);
    fill(cutpoint_config, builder.config.cutpoint);
    fill(stop_config, builder.config.stop);
    fill(binarize_config, builder.config.binarization);
    fill(grouping_config, builder.config.grouping);
    fill(leaf_config, builder.config.leaf);
  }

  nlohmann::json ModelParams::to_json() const {
    nlohmann::json j;

    if (!mode_input.empty()) {
      j["mode"] = mode_input;
    }
    j["size"]        = size;
    j["lambda"]      = lambda;
    j["max_retries"] = max_retries;

    if (seed) {
      j["seed"] = *seed;
    }
    if (threads) {
      j["threads"] = *threads;
    }
    if (n_vars) {
      j["n_vars"] = *n_vars;
    }
    if (p_vars) {
      j["p_vars"] = *p_vars;
    }

    if (!pp_config.is_null()) {
      j["pp"] = pp_config;
    }
    if (!vars_config.is_null()) {
      j["vars"] = vars_config;
    }
    if (!cutpoint_config.is_null()) {
      j["cutpoint"] = cutpoint_config;
    }
    if (!stop_config.is_null()) {
      j["stop"] = stop_config;
    }
    if (!binarize_config.is_null()) {
      j["binarize"] = binarize_config;
    }
    if (!grouping_config.is_null()) {
      j["grouping"] = grouping_config;
    }
    if (!leaf_config.is_null()) {
      j["leaf"] = leaf_config;
    }

    return j;
  }
}
