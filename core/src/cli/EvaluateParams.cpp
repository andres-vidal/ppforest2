/**
 * @file EvaluateParams.cpp
 * @brief Evaluate and simulate parameter construction and resolution.
 */
#include "cli/EvaluateParams.hpp"
#include "cli/Validation.hpp"
#include "serialization/JsonOptional.hpp"
#include "utils/UserError.hpp"

#include <fmt/format.h>
#include <regex>

namespace ppforest2::cli {

  SimulateParams::SimulateParams(nlohmann::json const& config) {
    format                         = config.value("simulate", format);
    classification.mean            = config.value("simulate_mean", classification.mean);
    classification.mean_separation = config.value("simulate_mean_separation", classification.mean_separation);
    regression.n_informative       = config.value("simulate_n_informative", regression.n_informative);
    regression.y_intercept         = config.value("simulate_y_intercept", regression.y_intercept);
    regression.y_sd                = config.value("simulate_y_sd", regression.y_sd);
    classification.sd              = config.value("simulate_sd", classification.sd);
    regression.sd                  = config.value("simulate_sd", regression.sd);
  }

  void SimulateParams::resolve_format() {
    if (format.empty()) {
      return;
    }

    static std::regex const pattern(R"((\d+)x(\d+)x(\d+))");
    std::smatch match;

    if (std::regex_match(format, match, pattern)) {
      rows     = std::stoi(match[1]);
      cols     = std::stoi(match[2]);
      n_groups = std::stoi(match[3]);
    }
  }

  stats::DataPacket SimulateParams::simulate(types::Mode mode, stats::RNG& rng) const {
    try {
      if (types::is_regression(mode)) {
        return stats::simulate(rows, cols, rng, regression);
      }
      return stats::simulate(rows, cols, n_groups, rng, classification);
    } catch (std::exception const& e) {
      throw UserError(fmt::format("Error simulating data: {}", e.what()));
    }
  }

  void EvaluateParams::validate(nlohmann::json const& config, std::vector<std::string>& errors) {
    // Train ratio
    if (config.contains("train_ratio") && config["train_ratio"].is_number()) {
      auto ratio = config["train_ratio"].get<float>();
      check(ratio > 0 && ratio < 1, "train_ratio must be in (0, 1)", errors);
    } else if (config.contains("train_ratio")) {
      errors.emplace_back("train_ratio must be a number");
    } else {
      errors.emplace_back("train_ratio is required");
    }

    // Iterations
    if (config.contains("iterations") && config["iterations"].is_number_integer()) {
      check(config["iterations"].get<int>() > 0, "iterations must be positive", errors);
    }

    // Convergence params
    if (config.contains("convergence") && config["convergence"].is_object()) {
      auto const& conv = config["convergence"];

      if (conv.contains("cv") && conv["cv"].is_number()) {
        auto cv = conv["cv"].get<float>();
        check(cv > 0 && cv <= 1, "convergence.cv must be in (0, 1]", errors);
      }

      if (conv.contains("min") && conv["min"].is_number_integer()) {
        check(conv["min"].get<int>() > 0, "convergence.min must be positive", errors);
      }

      if (conv.contains("max") && conv["max"].is_number_integer()) {
        check(conv["max"].get<int>() > 0, "convergence.max must be positive", errors);
      }

      if (conv.contains("window") && conv["window"].is_number_integer()) {
        check(conv["window"].get<int>() > 0, "convergence.window must be positive", errors);
      }
    }

    // Warmup
    if (config.contains("warmup") && config["warmup"].is_number_integer()) {
      check(config["warmup"].get<int>() >= 0, "warmup must be non-negative", errors);
    }
  }

  EvaluateParams::EvaluateParams(nlohmann::json const& config) {
    train_ratio = config.value("train_ratio", train_ratio);
    iterations  = config.value("iterations", iterations);
    warmup      = config.value("warmup", warmup);
    fixed_seed  = config.value("fixed_seed", fixed_seed);

    if (config.contains("convergence") && config["convergence"].is_object()) {
      auto const& conv   = config["convergence"];
      convergence.cv     = conv.value("cv", convergence.cv);
      convergence.min    = conv.value("min", convergence.min);
      convergence.max    = conv.value("max", convergence.max);
      convergence.window = conv.value("window", convergence.window);
    }
  }

  void EvaluateParams::resolve_defaults() {
    if (!train_ratio) {
      train_ratio = 0.7F;
    }
    if (!iterations) {
      iterations = 0;
    }
    if (!convergence.cv) {
      convergence.cv = 0.05F;
    }
    if (!convergence.min) {
      convergence.min = 10;
    }
    if (!convergence.max) {
      convergence.max = 200;
    }
    if (!convergence.window) {
      convergence.window = 3;
    }
  }

  nlohmann::json EvaluateParams::to_json() const {
    nlohmann::json j;

    if (train_ratio) {
      j["train_ratio"] = train_ratio.value();
    }
    if (iterations && iterations.value() > 0) {
      j["iterations"] = iterations.value();
    }

    j["warmup"] = warmup;

    if (fixed_seed) {
      j["fixed_seed"] = true;
    }

    nlohmann::json conv = nlohmann::json::object();
    if (convergence.cv) {
      conv["cv"] = convergence.cv.value();
    }
    if (convergence.min) {
      conv["min"] = convergence.min.value();
    }
    if (convergence.max) {
      conv["max"] = convergence.max.value();
    }
    if (convergence.window) {
      conv["window"] = convergence.window.value();
    }
    if (!conv.empty()) {
      j["convergence"] = conv;
    }

    return j;
  }

  nlohmann::json EvaluateParams::overrides() const {
    nlohmann::json j = nlohmann::json::object();

    if (iterations) {
      j["iterations"] = iterations.value();
    }
    if (train_ratio) {
      j["train_ratio"] = train_ratio.value();
    }

    nlohmann::json conv = nlohmann::json::object();

    if (convergence.cv) {
      conv["cv"] = convergence.cv.value();
    }
    if (convergence.min) {
      conv["min"] = convergence.min.value();
    }
    if (convergence.max) {
      conv["max"] = convergence.max.value();
    }
    if (convergence.window) {
      conv["window"] = convergence.window.value();
    }

    if (!conv.empty()) {
      j["convergence"] = conv;
    }

    return j;
  }
}
