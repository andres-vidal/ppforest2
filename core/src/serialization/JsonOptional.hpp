/**
 * @file JsonOptional.hpp
 * @brief nlohmann::json support for std::optional<T>.
 *
 * Include this header alongside <nlohmann/json.hpp> to enable
 * transparent JSON conversion for std::optional fields:
 *
 * @code
 *   j["key"] = std::optional<int>(42);   // writes 42
 *   j["key"] = std::optional<int>{};     // writes null
 *   auto opt = j["key"].get<std::optional<int>>();  // reads back
 * @endcode
 */
#pragma once

#include <nlohmann/json.hpp>
#include <optional>
#include <string_view>

namespace ppforest2::serialization {
  /**
   * @brief True iff @p j has a non-null value at @p key.
   *
   * Centralises the "key is present and not null" check used when
   * reading optional fields. Callers that previously wrote
   * `j.contains(key)` must now use this helper (or equivalent) to
   * tolerate the `null` that the writer emits for `std::nullopt`.
   */
  inline bool has_value(nlohmann::json const& j, std::string_view key) {
    auto it = j.find(key);
    return it != j.end() && !it->is_null();
  }
}

namespace nlohmann {
  template<typename T> struct adl_serializer<std::optional<T>> {
    static void to_json(json& j, std::optional<T> const& opt) {
      if (opt) {
        j = *opt;
      } else {
        j = nullptr;
      }
    }

    static void from_json(json const& j, std::optional<T>& opt) {
      if (j.is_null()) {
        opt = std::nullopt;
      } else {
        opt = j.get<T>();
      }
    }
  };
}
