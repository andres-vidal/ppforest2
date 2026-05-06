#pragma once

#include <initializer_list>
#include <limits>
#include <string>

#include <nlohmann/json.hpp>

namespace ppforest2 {
  /**
   * @brief A small DSL for extracting-and-validating values out of a JSON
   *        object with path-aware error messages.
   *
   * `JsonReader` wraps a `nlohmann::json` object and a dotted path (e.g.
   * `"config.pp"`). Every extraction method throws `std::runtime_error`
   * with a message that names the offending field:
   *
   *   "config.pp.lambda: expected number, got string"
   *   "config.pp: unexpected key 'lamda' (allowed: name, lambda)"
   *   "config.stop.min_size: must be in [2, ∞) (got 0)"
   *
   * Used both at the top level (see `serialization::validate_model_export`)
   * and inside every strategy's `from_json`. Unifies what `validate_json_keys`
   * + `j.at(k).get<T>()` used to do piecemeal.
   *
   * Scope is deliberately small:
   * - typed field extraction (`require<T>` / `optional<T>`)
   * - enum-like string validation (`require_enum`)
   * - numeric range validation (`require_number` / `require_int`)
   * - unknown-key rejection (`only_keys`)
   * - nested-object descent (`at`)
   * - array access (`require_array`)
   *
   * The DSL is not a schema validator and cannot express composition
   * (`oneOf`, `allOf`, recursive schemas). If you find yourself wanting
   * those, bring in a real schema library instead of growing this file.
   */
  class JsonReader {
  public:
    /**
     * @param j     The JSON object this reader wraps.
     * @param path  Dotted path used as the prefix for error messages
     *              (e.g. `"config.pp"`). The reader does not verify that
     *              `j` itself is an object at construction — use
     *              `require_object()` explicitly for that.
     */
    JsonReader(nlohmann::json const& j, std::string path);

    /** @brief Assert the wrapped value is a JSON object. */
    void require_object() const;

    /**
     * @brief Descend into a required sub-object.
     *
     * @throws std::runtime_error if `key` is missing or not an object.
     */
    JsonReader at(std::string const& key) const;

    /**
     * @brief Extract a required typed value.
     *
     * @tparam T       Target type (e.g. `int`, `float`, `std::string`).
     * @param key      Field name within the wrapped object.
     * @throws         `std::runtime_error` with the dotted path on
     *                 missing keys or type mismatch.
     */
    template<typename T> T require(std::string const& key) const;

    /** @brief Extract an optional typed value, falling back to @p fallback. */
    template<typename T> T optional(std::string const& key, T fallback) const;

    /** @brief Whether @p key is present on the wrapped object. */
    bool contains(std::string const& key) const;

    /**
     * @brief Extract a required string constrained to @p allowed values.
     *
     * @throws std::runtime_error if the key is missing, not a string, or
     *         the value is not in the allowed set.
     */
    std::string require_enum(std::string const& key, std::initializer_list<char const*> allowed) const;

    /**
     * @brief Extract a required integer value, optionally constrained.
     *
     * Accepts any JSON number whose value is an integer — both literals
     * written without a fractional part (`5`) and equivalent numeric
     * forms (`5.0`, `5e0`). Non-integer fractions (`5.5`), non-finite
     * numbers (`NaN`, `Inf`), and non-numeric types are rejected.
     *
     * This matches JSON Schema's `type: integer` semantics, which is
     * *value-based*: the spec (RFC 8259 §6) does not distinguish integer
     * from non-integer JSON numbers at the format level, and JSON Schema
     * defines `integer` as "any JSON number with zero fractional part."
     * nlohmann's `is_number_integer()` is a parse-level tag (whether the
     * literal had a decimal point) that's stricter than the spec — we
     * deliberately don't inherit it. This means values round-tripped
     * through languages that serialize all numbers uniformly (R, Python
     * with default `json.dumps`) validate correctly.
     *
     * @param key  Field name within the wrapped object.
     * @param min  Inclusive lower bound (default: no lower bound).
     * @param max  Inclusive upper bound (default: no upper bound).
     */
    long long require_int(
        std::string const& key,
        long long min = std::numeric_limits<long long>::min(),
        long long max = std::numeric_limits<long long>::max()
    ) const;

    /** @brief Extract a required number (double), optionally constrained. */
    double require_number(
        std::string const& key,
        double min = -std::numeric_limits<double>::infinity(),
        double max = std::numeric_limits<double>::infinity()
    ) const;

    /**
     * @brief Assert the wrapped object contains only the given keys.
     *
     * @throws std::runtime_error naming the first unexpected key it finds
     *         (plus the list of allowed keys for context).
     */
    void only_keys(std::initializer_list<char const*> allowed) const;

    /**
     * @brief Assert every key in @p keys is present on the wrapped object.
     *
     * Mirror of `only_keys`: where that one rejects extra keys, this one
     * rejects missing ones. Throws on the first absent key it finds.
     */
    void require_keys(std::initializer_list<char const*> keys) const;

    /**
     * @brief Return a const reference to a required array field.
     *
     * @throws std::runtime_error if the key is missing or the value is
     *         not a JSON array.
     */
    nlohmann::json const& require_array(std::string const& key) const;

    /**
     * @brief Assert @p key is an array of strings, optionally non-empty.
     *
     * @throws std::runtime_error if the key is missing, not an array,
     *         empty (when @p non_empty is true), or contains non-string
     *         elements (path includes `[i]` for the offending index).
     */
    void require_string_array(std::string const& key, bool non_empty = false) const;

    /**
     * @brief Throw if @p key is present and non-null on the wrapped object.
     *
     * Mode-homogeneous schemas use this to express "this field must be
     * absent / null in this variant" — e.g. classification's `meta.groups`
     * is required, regression's must be null.
     */
    void forbid_key(std::string const& key, std::string const& reason) const;

    /** @name Discard-result siblings of `require_*` for validators that don't need the value. */
    ///@{
    void expect_enum(std::string const& key, std::initializer_list<char const*> allowed) const {
      (void)require_enum(key, allowed);
    }
    void expect_int(
        std::string const& key,
        long long min = std::numeric_limits<long long>::min(),
        long long max = std::numeric_limits<long long>::max()
    ) const {
      (void)require_int(key, min, max);
    }
    void expect_number(
        std::string const& key,
        double min = -std::numeric_limits<double>::infinity(),
        double max = std::numeric_limits<double>::infinity()
    ) const {
      (void)require_number(key, min, max);
    }
    void expect_array(std::string const& key) const { (void)require_array(key); }
    ///@}

    /** @brief The raw underlying JSON (escape hatch for unusual cases). */
    nlohmann::json const& json() const { return j_; }

    /** @brief The path prefix used in error messages. */
    std::string const& path() const { return path_; }

  private:
    nlohmann::json const& j_;
    std::string path_;

    std::string child_path(std::string const& key) const;
    nlohmann::json const& require_present(std::string const& key) const;
  };
}
