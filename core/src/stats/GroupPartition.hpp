#pragma once


#include "stats/Stats.hpp"
#include "utils/Types.hpp"
#include "utils/Map.hpp"
#include "utils/Invariant.hpp"

#include <map>
#include <optional>
#include <set>
#include <vector>
#include <Eigen/Dense>

namespace ppforest2::stats {
  /**
   * @brief Contiguous-block representation of grouped observations.
   *
   * Assumes the response vector is sorted so that observations of the
   * same class are contiguous.  Stores the start/end row indices of
   * each class block and provides efficient extraction, subsetting,
   * and computation of between- and within-group statistics.
   *
   * Groups can be hierarchically merged via remap(), which assigns
   * supergroup labels while tracking the original subgroups.
   *
   * @code
   *   // y must be sorted so equal values are contiguous.
   *   GroupPartition gp(y);
   *
   *   // Extract rows belonging to group 0:
   *   auto x_group0 = gp.group(x, 0);
   *
   *   // Between- and within-group statistics:
   *   auto B = gp.bgss(x);   // between-group sum of squares (p × p)
   *   auto W = gp.wgss(x);   // within-group sum of squares  (p × p)
   *
   *   // Restrict to a subset of groups:
   *   GroupPartition sub = gp.subset({0, 2});
   * @endcode
   */
  class GroupPartition {
    using Group       = types::Response;
    using GroupSet    = std::set<types::Response>;
    using GroupMap    = std::map<types::Response, types::Response>;
    using GroupInvMap = std::map<types::Response, GroupSet>;
    using GroupVector = types::Vector<types::Response>;

    public:
      /** @brief Check whether all equal values in @p y form a single contiguous block. */
      static bool is_contiguous(const GroupVector& y);

      /**
       * @brief Construct from a sorted response vector.
       *
       * @param y  Response vector (n) with contiguous class blocks.
       */
      GroupPartition(const types::ResponseVector& y);

      /** @brief First row index of the block for @p group. */
      int group_start(const Group& group) const;
      /** @brief Last row index (inclusive) of the block for @p group. */
      int group_end(const Group& group) const;
      /** @brief Number of observations in @p group. */
      int group_size(const Group& group) const;

      /**
       * @brief Extract rows belonging to a group (or supergroup).
       *
       * @param x      Feature matrix (n × p).
       * @param group  Group label.
       * @return       Sub-matrix of rows belonging to @p group.
       */
      auto group(const types::FeatureMatrix& x, const Group& group) const {
        std::vector<int> indices;

        const auto& subs = this->subgroups.at(group);

        for (const auto& g : subs) {
          for (int i = group_start(g); i <= group_end(g); ++i) {
            invariant(i >= 0 && i < x.rows(),
              "GroupPartition::group: index out of bounds");
            indices.push_back(i);
          }
        }

        return x(indices, Eigen::placeholders::all);
      }

      /**
       * @brief Extract all rows across all groups.
       *
       * @param x  Feature matrix (n × p).
       * @return   Sub-matrix with all grouped rows.
       */
      auto data(const types::FeatureMatrix& x) const {
        std::vector<int> indices;

        for (const auto& kv : Blocks) {
          const auto& g = kv.first;
          for (int i = group_start(g); i <= group_end(g); ++i) {
            indices.push_back(i);
          }
        }

        return x(indices, Eigen::placeholders::all);
      }

      /** @brief Overall mean of all grouped rows (p). */
      types::FeatureVector mean(const types::FeatureMatrix& x) const;
      /** @brief Between-group sum of squares matrix (p × p). */
      types::FeatureMatrix bgss(const types::FeatureMatrix& x) const;
      /** @brief Within-group sum of squares matrix (p × p). */
      types::FeatureMatrix wgss(const types::FeatureMatrix& x) const;

      /**
       * @brief Create a partition containing only the given groups.
       *
       * @param groups  Set of group labels to keep.
       * @return        New GroupPartition restricted to @p groups.
       */
      GroupPartition subset(GroupSet groups) const;

      /**
       * @brief Merge groups according to a mapping.
       *
       * @param mapping  Maps original group labels to supergroup labels.
       * @return         New GroupPartition with merged groups.
       */
      GroupPartition remap(const GroupMap& mapping) const;

      /** @brief Set of all group labels in this partition. */
      const GroupSet groups;
      /** @brief Maps each group to its supergroup (identity if no merge). */
      const GroupMap supergroups;
      /** @brief Maps each group to its set of subgroups. */
      const GroupInvMap subgroups;

    private:
      struct Block {
        int start;
        int end;
        int size;
        std::optional<types::Response> next;
        std::optional<types::Response> prev;
      };

      using BlockMap = std::map<types::Response, Block>;
      const BlockMap Blocks;

      BlockMap init_Blocks(const GroupVector& y);
      GroupMap init_supergroups();

      GroupPartition(
        const BlockMap& Blocks,
        const GroupSet& groups);

      GroupPartition(
        const BlockMap& Blocks,
        const GroupMap& supergroups);

      GroupPartition(
        const BlockMap& Blocks,
        const GroupSet& groups,
        const GroupMap& supergroups);
  };
}
