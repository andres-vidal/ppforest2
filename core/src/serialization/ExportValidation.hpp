#pragma once

#include <nlohmann/json.hpp>

namespace ppforest2::serialization {
  /**
   * @brief Validate the structural shape of a loaded model Export JSON.
   *
   * Runs before any `from_json` decoding so failures surface with
   * domain-specific, path-annotated messages instead of nlohmann's
   * cryptic type errors.
   *
   * Concentrates every cross-field and required-field check in one
   * place. Strategy internals are *not* validated here — each strategy's
   * own `from_json` performs that via `JsonReader`. This keeps the
   * validator responsibilities flat (skeleton + `config` + `meta` +
   * variant-specific model block) and avoids turning it into a schema
   * library.
   *
   * Each validator asserts `model_type == "tree"` / `"forest"` matches
   * the call site; mismatches throw a `UserError`.
   *
   * @throws UserError with a dotted path on validation failure.
   */
  void validate_tree_export(nlohmann::json const& j);

  /** @copydoc validate_tree_export */
  void validate_forest_export(nlohmann::json const& j);
}
