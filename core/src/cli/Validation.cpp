/**
 * @file Validation.cpp
 * @brief Central config validation for all CLI subcommands.
 */
#include "cli/Validation.hpp"
#include "cli/CLIOptions.hpp"
#include "utils/UserError.hpp"

#include <fmt/format.h>
#include <optional>
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
          // The regex guarantees digits, but not that they fit in an int —
          // std::stoi throws out_of_range on oversized dimensions.
          auto to_int = [](std::string const& s) -> std::optional<int> {
            try {
              return std::stoi(s);
            } catch (std::exception const&) {
              return std::nullopt;
            }
          };

          auto const n = to_int(match[1]);
          auto const p = to_int(match[2]);
          auto const g = to_int(match[3]);

          if (!n || !p || !g) {
            errors.emplace_back("simulate: dimensions must fit in a 32-bit integer");
          } else {
            check(*n > 0, "simulate: n must be positive", errors);
            check(*p > 0, "simulate: p must be positive", errors);
            check(*g > 1, "simulate: g must be > 1", errors);
          }
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
