#include "serialization/ExportValidation.hpp"

#include "utils/JsonReader.hpp"
#include "utils/UserError.hpp"

#include <string>

namespace ppforest2::serialization {
  namespace {
    // Top-level is deliberately an open set. The writer emits the core
    // `model`/`config`/`meta` trio plus optional metrics (confusion
    // matrices, regression metrics, OOB error, VI) and CLI annotations
    // (`training_duration_ms`, `save_path`). Downstream tools (e.g. the
    // golden generator) add more annotations on top. Enforcing a closed
    // set would reject any such extension. Required-key presence is
    // still enforced below.

    constexpr auto MODES = {"classification", "regression"};

    constexpr auto CONFIG_REQUIRED_KEYS = {
        "mode",
        "size",
        "seed",
        "threads",
        "max_retries",
        "pp",
        "vars",
        "cutpoint",
        "stop",
        "binarize",
        "grouping",
        "leaf"
    };

    // Validate the `config` block, returning the mode string. `binarize` is
    // required for every mode — regression specs carry `binarize::Disabled`
    // (a mode-agnostic placeholder) rather than omitting the key.
    std::string validate_config(JsonReader const& config) {
      config.require_keys(CONFIG_REQUIRED_KEYS);
      config.expect_int("size", 0);
      config.expect_int("seed");
      config.expect_int("threads", 0);
      config.expect_int("max_retries", 0);
      return config.require_enum("mode", MODES);
    }

    // Validate the `meta` block, cross-checking against the config mode.
    //
    // The schema is mode-homogeneous: `meta.groups` is always present, with
    // shape dictated by `meta.mode`. For regression the value is `null`
    // (no group concept); for classification it must be a non-empty array
    // of strings. See `Export<Model::Ptr>::to_json` for the writer side.
    void validate_meta(JsonReader const& meta, std::string const& config_mode) {
      meta.expect_int("observations", 0);
      meta.expect_int("features", 0);

      // `feature_names` is always written by `Export<Model::Ptr>::to_json` —
      // populated from the source CSV header, or empty when training paths
      // (simulated data, R bindings) don't carry column names.
      meta.require_string_array("feature_names");

      // `groups` is always emitted as an array — empty for regression (no
      // group concept), non-empty for classification. The discriminator
      // is `config.mode`, not the array's emptiness.
      bool const is_regression = config_mode == "regression";
      meta.require_string_array("groups", /*non_empty=*/!is_regression);
    }

    // Shared skeleton check (config + meta). Variant-specific `model`
    // shape is checked by the model-block validators below.
    JsonReader validate_skeleton(nlohmann::json const& j) {
      JsonReader root(j, "Export");
      root.require_object();
      validate_meta(root.at("meta"), validate_config(root.at("config")));
      return root;
    }

    void validate_tree_model_block(JsonReader const& root) {
      root.at("model").require_keys({"root"});
    }

    void validate_forest_model_block(JsonReader const& root) {
      root.at("model").expect_array("trees");
    }

    // Tree/forest validators share a "model_type must equal the expected
    // variant" check.
    void expect_model_type(JsonReader const& root, char const* expected) {
      std::string const t = root.require_enum("model_type", {"tree", "forest"});
      user_error(t == expected, "Export.model_type: expected '" + std::string(expected) + "' (got '" + t + "')");
    }
  }

  void validate_tree_export(nlohmann::json const& j) {
    auto const root = validate_skeleton(j);
    expect_model_type(root, "tree");
    validate_tree_model_block(root);
  }

  void validate_forest_export(nlohmann::json const& j) {
    auto const root = validate_skeleton(j);
    expect_model_type(root, "forest");
    validate_forest_model_block(root);
  }
}
