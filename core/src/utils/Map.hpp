#pragma once

#include <map>
#include <set>

/** @brief Utility functions for std::map manipulation. */
namespace ppforest2::utils {
  /**
   * @brief Invert a map: values become keys, original keys are grouped into sets.
   *
   * @param map  Input map {K → V}.
   * @return     Inverted map {V → set<K>}.
   */
  template<typename K, typename V> std::map<V, std::set<K>> invert(std::map<K, V> const& map) {
    std::map<V, std::set<K>> result;

    for (auto const& [key, value] : map) {
      result[value].insert(key);
    }

    return result;
  }

  /**
   * @brief Extract all keys from a map as a set.
   *
   * @param map  Input map.
   * @return     Set of all keys.
   */
  template<typename K, typename V> std::set<K> keys(std::map<K, V> const& map) {
    std::set<K> result;

    for (auto const& [key, value] : map) {
      result.insert(key);
    }

    return result;
  }

  /**
   * @brief Extract all values from a map as a set.
   *
   * @param map  Input map.
   * @return     Set of all values (duplicates collapsed).
   */
  template<typename K, typename V> std::set<V> values(std::map<K, V> const& map) {
    std::set<V> result;

    for (auto const& [key, value] : map) {
      result.insert(value);
    }

    return result;
  }
}
