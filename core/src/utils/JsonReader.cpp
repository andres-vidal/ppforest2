#include "utils/JsonReader.hpp"

#include "serialization/JsonOptional.hpp"
#include "utils/UserError.hpp"

#include <cmath>
#include <limits>
#include <sstream>
#include <string_view>

namespace ppforest2 {
  namespace {
    /**
     * @brief Comma-join an allowed-keys / allowed-values list for error messages.
     *
     * Shared between `only_keys` and `require_enum`, which both build
     * messages of the form `"allowed: a, b, c"`.
     */
    std::string join_allowed(std::initializer_list<char const*> allowed) {
      std::string out;
      bool first = true;
      for (char const* a : allowed) {
        if (!first) {
          out += ", ";
        }
        first = false;
        out += a;
      }
      return out;
    }

    /**
     * @brief Throw a `UserError` prefixed with the dotted path.
     *
     * All JSON-validation failures funnel through here so error messages
     * share one format: `"<path>: <what>"`. Validation failures are caused
     * by user-provided JSON, not programmer bugs — so callers can catch
     * `UserError` to surface actionable messages (and let real bugs
     * propagate as `runtime_error` / `logic_error` instead).
     */
    [[noreturn]] void throw_error(std::string const& path, std::string const& what) {
      throw UserError(path + ": " + what);
    }

    /**
     * @brief Throw a path-prefixed "expected <expected>, got <type_name>"
     * — every shape-check method goes through this.
     */
    [[noreturn]] void throw_type_error(std::string const& path, char const* expected, nlohmann::json const& got) {
      throw_error(path, std::string("expected ") + expected + ", got " + got.type_name());
    }

    /** @brief Linear lookup used by `only_keys` and `require_enum`. */
    bool is_one_of(std::string_view value, std::initializer_list<char const*> allowed) {
      for (char const* a : allowed) {
        if (value == a) {
          return true;
        }
      }
      return false;
    }

    /**
     * @brief Format a `"must be in [<min>, <max>] (got <v>)"` message,
     * substituting `-∞`/`∞` for the sentinel min/max values.
     */
    template<typename T> std::string format_range_error(T min, T max, T got, T lower_sentinel, T upper_sentinel) {
      std::ostringstream oss;
      oss << "must be in [";
      if (min == lower_sentinel) {
        oss << "-∞";
      } else {
        oss << min;
      }
      oss << ", ";
      if (max == upper_sentinel) {
        oss << "∞";
      } else {
        oss << max;
      }
      oss << "] (got " << got << ")";
      return oss.str();
    }

    /**
     * @brief Extract a `long long` from a JSON value.
     *
     * Accepts integer literals and integer-valued floats (R serializes its
     * `numeric` type without an integer marker, so `5L` round-trips as
     * `5.0`). Rejects non-finite floats, fractional floats, and floats
     * outside the representable `long long` range.
     */
    long long extract_integer(nlohmann::json const& v, std::string const& path) {
      if (v.is_number_integer()) {
        return v.get<long long>();
      }
      if (v.is_number_float()) {
        double const d = v.get<double>();
        if (!std::isfinite(d) || d != std::floor(d)) {
          throw_error(path, "expected integer, got non-integer number (" + v.dump() + ")");
        }
        // `(double)LLONG_MAX` rounds up (LLONG_MAX is 2^63 - 1 but the closest
        // representable double is 2^63), so use strict comparisons against the
        // next representable powers-of-two to stay away from the UB edge of
        // `static_cast<long long>(d)` for out-of-range values.
        constexpr double kMin = -9.2233720368547758e18; // -2^63
        constexpr double kMax = 9.2233720368547758e18;  //  2^63
        if (d <= kMin || d >= kMax) {
          throw_error(path, "integer out of representable range (got " + v.dump() + ")");
        }
        return static_cast<long long>(d);
      }
      throw_type_error(path, "integer", v);
    }
  }

  JsonReader::JsonReader(nlohmann::json const& j, std::string path)
      : j_(j)
      , path_(std::move(path)) {}

  /**
   * @brief Compose the dotted path for a child key.
   *
   * Returns `"<path_>.<key>"`, or just `key` when the current path is empty
   * (i.e. the reader was constructed at the document root). Keys with dots
   * in them are not quoted — the model JSON doesn't use them.
   */
  std::string JsonReader::child_path(std::string const& key) const {
    return path_.empty() ? key : path_ + "." + key;
  }

  void JsonReader::require_object() const {
    if (!j_.is_object()) {
      throw_type_error(path_, "object", j_);
    }
  }

  /**
   * @brief Look up a key that must exist, returning the raw JSON value.
   *
   * Asserts the wrapped value is a JSON object, then errors on missing
   * key. Shared entry point used by every `require_*` method to localise
   * the "key missing" error message.
   */
  nlohmann::json const& JsonReader::require_present(std::string const& key) const {
    require_object();
    auto it = j_.find(key);
    if (it == j_.end()) {
      throw_error(child_path(key), "missing required key");
    }
    return *it;
  }

  JsonReader JsonReader::at(std::string const& key) const {
    auto const& child = require_present(key);
    if (!child.is_object()) {
      throw_type_error(child_path(key), "object", child);
    }
    return JsonReader(child, child_path(key));
  }

  bool JsonReader::contains(std::string const& key) const {
    return j_.is_object() && j_.contains(key);
  }

  template<typename T> T JsonReader::require(std::string const& key) const {
    auto const& v = require_present(key);
    try {
      return v.get<T>();
    } catch (nlohmann::json::type_error const&) {
      throw_error(child_path(key), std::string("unexpected type: ") + v.type_name());
    }
  }

  // cppcheck-suppress passedByValue
  // `fallback` is by value so it can be moved into the return on the
  // missing-key path; `T const&` would force a copy-construct at the
  // return site with no perf benefit.
  template<typename T> T JsonReader::optional(std::string const& key, T fallback) const {
    if (!contains(key)) {
      return fallback;
    }
    return require<T>(key);
  }

  std::string JsonReader::require_enum(std::string const& key, std::initializer_list<char const*> allowed) const {
    auto const& v = require_present(key);
    if (!v.is_string()) {
      throw_type_error(child_path(key), "string", v);
    }
    std::string const value = v.get<std::string>();
    if (!is_one_of(value, allowed)) {
      throw_error(child_path(key), "must be one of [" + join_allowed(allowed) + "] (got '" + value + "')");
    }
    return value;
  }

  long long JsonReader::require_int(std::string const& key, long long min, long long max) const {
    long long const value = extract_integer(require_present(key), child_path(key));
    if (value < min || value > max) {
      throw_error(
          child_path(key),
          format_range_error(
              min, max, value, std::numeric_limits<long long>::min(), std::numeric_limits<long long>::max()
          )
      );
    }
    return value;
  }

  double JsonReader::require_number(std::string const& key, double min, double max) const {
    auto const& v = require_present(key);
    if (!v.is_number()) {
      throw_type_error(child_path(key), "number", v);
    }
    double const value = v.get<double>();
    if (value < min || value > max) {
      throw_error(
          child_path(key),
          format_range_error(
              min, max, value, -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()
          )
      );
    }
    return value;
  }

  void JsonReader::only_keys(std::initializer_list<char const*> allowed) const {
    require_object();
    for (auto it = j_.begin(); it != j_.end(); ++it) {
      if (!is_one_of(it.key(), allowed)) {
        throw_error(path_, "unexpected key '" + it.key() + "' (allowed: " + join_allowed(allowed) + ")");
      }
    }
  }

  void JsonReader::require_keys(std::initializer_list<char const*> keys) const {
    require_object();
    for (char const* k : keys) {
      if (!j_.contains(k)) {
        throw_error(child_path(k), "missing required key");
      }
    }
  }

  nlohmann::json const& JsonReader::require_array(std::string const& key) const {
    auto const& v = require_present(key);
    if (!v.is_array()) {
      throw_type_error(child_path(key), "array", v);
    }
    return v;
  }

  void JsonReader::require_string_array(std::string const& key, bool non_empty) const {
    auto const& arr = require_array(key);
    if (non_empty && arr.empty()) {
      throw_error(child_path(key), "must be non-empty");
    }
    for (std::size_t i = 0; i < arr.size(); ++i) {
      if (!arr[i].is_string()) {
        throw_type_error(child_path(key) + "[" + std::to_string(i) + "]", "string", arr[i]);
      }
    }
  }

  void JsonReader::forbid_key(std::string const& key, std::string const& reason) const {
    if (serialization::has_value(j_, key)) {
      throw_error(child_path(key), reason);
    }
  }

  // Explicit instantiations for the types we actually use.
  template int JsonReader::require<int>(std::string const&) const;
  template long long JsonReader::require<long long>(std::string const&) const;
  template float JsonReader::require<float>(std::string const&) const;
  template double JsonReader::require<double>(std::string const&) const;
  template std::string JsonReader::require<std::string>(std::string const&) const;
  template bool JsonReader::require<bool>(std::string const&) const;

  template int JsonReader::optional<int>(std::string const&, int) const;
  template long long JsonReader::optional<long long>(std::string const&, long long) const;
  template float JsonReader::optional<float>(std::string const&, float) const;
  template double JsonReader::optional<double>(std::string const&, double) const;
  template std::string JsonReader::optional<std::string>(std::string const&, std::string) const;
  template bool JsonReader::optional<bool>(std::string const&, bool) const;
}
