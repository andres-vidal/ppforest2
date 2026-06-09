/**
 * @file ModelParams.hpp
 * @brief Model training parameters shared by train and evaluate.
 */
#pragma once

#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

namespace ppforest2::cli {
  /** @brief Model training parameters shared by train and evaluate. */
  struct ModelParams {
    ModelParams() = default;

    int size     = 100;
    float lambda = 0.0;
    std::optional<int> threads;
    std::optional<int> seed;
    std::optional<float> p_vars;
    std::optional<int> n_vars;
    int max_retries = 3;
    std::string p_vars_input;

    /**
     * @brief Training mode ("classification" or "regression").
     *
     * Empty default = "infer from data": for `--data`, the CSV reader
     * detects regression from fractional y values and `read_data`
     * populates this; for `--simulate`, no data exists yet so we fall
     * back to "classification". `-m,--mode` overrides both paths.
     */
    std::string mode_input;

    /** @brief Explicit strategy inputs (--X flags). */
    std::string pp_input;
    std::string vars_input;
    std::string cutpoint_input;
    /**
     * @brief Repeated `--stop` flag values.
     *
     * The only strategy flag that accepts repeats: multiple occurrences
     * collect into this vector and are composed into a `CompositeStop`
     * `any`-rule at `resolve()` time. A single `--stop` behaves unchanged.
     */
    std::vector<std::string> stop_inputs;
    std::string binarize_input;
    std::string grouping_input;
    std::string leaf_input;

    /** @brief Strategy JSON objects (from CLI strings or config file). */
    nlohmann::json pp_config;
    nlohmann::json vars_config;
    nlohmann::json cutpoint_config;
    nlohmann::json stop_config;
    nlohmann::json binarize_config;
    nlohmann::json grouping_config;
    nlohmann::json leaf_config;

    /** @brief Construct from a JSON config object. */
    explicit ModelParams(nlohmann::json const& config);

    /** @brief Parse strategy input strings and p_vars input into their final form. */
    void resolve();

    /** @brief Fill in runtime defaults (threads, vars, strategy configs). */
    void resolve_defaults(unsigned int total_vars);

    /** @brief Validate model-related fields in a JSON config. */
    static void validate(nlohmann::json const& config, std::vector<std::string>& errors);

    /** @brief Serialize to JSON config. */
    nlohmann::json to_json() const;
  };

  /**
   * @brief Parse a proportion from a string or JSON value.
   *
   * Formats:
   * - String: "1/3", "0.5"
   * - JSON string: "1/3", "0.5"
   * - JSON number: 0.5, 1.0
   *
   * @return A value in (0, 1].
   * @throws UserError on invalid input.
   */
  float parse_proportion(std::string const& input);
  float parse_proportion(nlohmann::json const& j);

  /**
   * @brief Parse a CLI strategy string into a JSON object.
   *
   * Converts e.g. `"pda:lambda=0.3"` to `{"name": "pda", "lambda": 0.3}`.
   */
  nlohmann::json strategy_string_to_json(std::string const& input);
}
