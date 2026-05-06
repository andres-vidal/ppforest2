#pragma once

#include "stats/Stats.hpp"
#include "utils/Types.hpp"

#include <set>
#include <string>
#include <vector>

namespace ppforest2::stats {
  /**
   * @brief Bundled dataset: features, response, and group labels.
   *
   * Convenience struct that groups a feature matrix and a response vector
   * with dataset-level metadata (unique group labels, column names).  Used
   * primarily for passing data through the training pipeline.
   *
   */
  struct DataPacket {
    /** @brief Feature matrix (n × p). */
    types::FeatureMatrix x;
    /** @brief Response vector (n) — integer labels (classification) or continuous response (regression). */
    types::OutcomeVector y;
    /** @brief Set of distinct group labels (classification only; empty for regression). */
    std::set<types::GroupId> groups;
    /**
     * @brief Original group label names, indexed by integer code.
     *
     * When populated, group_names[i] is the original string label
     * that maps to integer code i.  Empty when data is not read
     * from CSV (e.g., simulated data) or for regression.
     */
    types::Names group_names;
    /**
     * @brief Original feature column names from the CSV header.
     *
     * When populated, feature_names[j] is the header label for
     * column j of x.  Empty when data is simulated.
     */
    types::Names feature_names;

    DataPacket(
        types::FeatureMatrix const& x,
        types::OutcomeVector const& y,
        std::set<types::GroupId> const& groups,
        types::Names const& group_names   = {},
        types::Names const& feature_names = {}
    )
        : x(x)
        , y(y)
        , groups(groups)
        , group_names(group_names)
        , feature_names(feature_names) {}

    DataPacket(
        types::FeatureMatrix const& x,
        types::OutcomeVector const& y,
        types::Names const& group_names   = {},
        types::Names const& feature_names = {}
    )
        : x(x)
        , y(y)
        , groups(unique(y.cast<types::GroupId>()))
        , group_names(group_names)
        , feature_names(feature_names) {}

    DataPacket() = default;
  };
}
