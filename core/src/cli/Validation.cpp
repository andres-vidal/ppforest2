/**
 * @file Validation.cpp
 * @brief Central config validation for all CLI subcommands.
 */
#include "cli/Validation.hpp"
#include "cli/CLIOptions.hpp"
#include "utils/UserError.hpp"

#include <fmt/format.h>
#include <regex>

namespace ppforest2::cli {
  nlohmann::json training_defaults() {
    return {
        {"size", 100},
        {"lambda", 0.0},
        {"train_ratio", 0.7},
        {"max_retries", 3},
    };
  }

  namespace {
    void validate_data_source(nlohmann::json const& config, std::vector<std::string>& errors) {
      bool has_data     = !config.value("data", "").empty();
      bool has_simulate = !config.value("simulate", "").empty();

      check(has_data || has_simulate, "data source is required (data or simulate)", errors);

      if (has_simulate) {
        static std::regex const pattern(R"((\d+)x(\d+)x(\d+))");
        std::string sim = config["simulate"].get<std::string>();
        std::smatch match;

        if (!std::regex_match(sim, match, pattern)) {
          errors.emplace_back("simulate format must be NxPxG (e.g. 1000x10x2)");
        } else {
          check(std::stoi(match[1]) > 0, "simulate: n must be positive", errors);
          check(std::stoi(match[2]) > 0, "simulate: p must be positive", errors);
          check(std::stoi(match[3]) > 1, "simulate: g must be > 1", errors);
        }
      }
    }
  }

  void validate_training_config(nlohmann::json const& config, std::vector<std::string>& errors) {
    validate_data_source(config, errors);
    ModelParams::validate(config, errors);
    EvaluateParams::validate(config, errors);
  }

  void validate_params(Params const& params) {
    std::vector<std::string> errors;
    auto config = params.to_json();
    validate_training_config(config, errors);

    if (!errors.empty()) {
      std::string msg;
      for (auto const& e : errors) {
        msg += e + "\n";
      }
      throw ppforest2::UserError(msg);
    }
  }
}
