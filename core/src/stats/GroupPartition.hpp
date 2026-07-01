#pragma once


#include "utils/Types.hpp"
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
   * same group are contiguous.  Stores the start/end row indices of
   * each group block and provides efficient extraction, subsetting,
   * and computation of between- and within-group statistics.
   *
   * Groups can be hierarchically merged via remap(), which assigns
   * supergroup labels while tracking the original subgroups.
   *
   * @code
   *   // y must be sorted so equal values are contiguous.
   *   GroupPartition y_part(y);
   *
   *   // Extract rows belonging to group 0:
   *   auto x_group0 = y_part.group(x, 0);
   *
   *   // Between- and within-group statistics:
   *   auto B = y_part.bgss(x);   // between-group sum of squares (p × p)
   *   auto W = y_part.wgss(x);   // within-group sum of squares  (p × p)
   *
   *   // Restrict to a subset of groups:
   *   GroupPartition sub = y_part.subset({0, 2});
   * @endcode
   */
  class GroupPartition {
    using Group       = types::GroupId;
    using GroupSet    = std::set<types::GroupId>;
    using GroupMap    = std::map<types::GroupId, types::GroupId>;
    using GroupInvMap = std::map<types::GroupId, GroupSet>;
    using GroupVector = types::GroupIdVector;

  public:
    /** @brief Check whether all equal values in @p y form a single contiguous block. */
    static bool is_contiguous(GroupVector const& y);

    /**
       * @brief Construct from a sorted response vector.
       *
       * @param y  Outcome vector (n) with contiguous group blocks.
       */
    explicit GroupPartition(types::GroupIdVector const& y);

    /**
     * @brief Construct from a float-typed response vector.
     *
     * Classification `y` is carried as `OutcomeVector` (float) throughout the
     * training pipeline; this overload casts to integer labels internally
     * before building the block map. Values must encode integer labels.
     */
    explicit GroupPartition(types::OutcomeVector const& y);

    /**
     * @brief Construct a single-group partition covering rows `[start, end]`.
     *
     * Group label is `0`. Use `bisect(mid)` to split into a 2-group partition.
     */
    GroupPartition(int start, int end);

    /**
     * @brief Bisect a single-group partition at row index @p mid into two groups.
     *
     * Group 0 covers `[start, mid - 1]`, group 1 covers `[mid, end]`. The
     * receiver must currently be a single-group partition (typically built
     * via the `(int, int)` ctor). Distinct from `split(SplitSizes)` below,
     * which subsets a multi-group partition along its existing structure.
     *
     * @throws via invariant if the partition has more than one group, or
     *         if @p mid is outside `(start, end]`.
     */
    GroupPartition bisect(int mid) const;

    /** @brief First row index of the block for @p group. */
    int group_start(Group const& group) const;
    /** @brief Last row index (inclusive) of the block for @p group. */
    int group_end(Group const& group) const;
    /** @brief Number of observations in @p group. */
    int group_size(Group const& group) const;

    /**
     * @brief Smallest group label in the partition.
     *
     * `groups` is a `std::set<GroupId>` so iteration is in ascending key
     * order; this returns the first such label. Caller must ensure the
     * partition is non-empty.
     */
    Group first_group() const {
      invariant(!groups.empty(), "GroupPartition::first_group: partition is empty");
      return *groups.begin();
    }

    /**
     * @brief Total number of observations across all groups in the partition.
     *
     * Used by the tree builder to detect "no-progress" grouping splits:
     * if a child partition covers the same row count as its parent, the
     * split failed to partition the data and the builder converts the node
     * to a leaf to avoid unbounded recursion.
     */
    int total_size() const;

    /**
       * @brief Extract rows belonging to a group (or supergroup).
       *
       * Returns an Eigen block expression (zero-copy view) into @p x.
       * The result must be consumed immediately or assigned to a
       * concrete matrix — do not store it in `auto` across statements.
       *
       * @param x      Feature matrix (n × p).
       * @param group  Group label.
       * @return       Block expression over the rows of @p group.
       */
    template<typename Derived> auto group(Eigen::MatrixBase<Derived> const& x, Group const& group) const {
      std::vector<int> indices;

      auto const& subs = this->subgroups.at(group);

      for (auto const& g : subs) {
        for (int i = group_start(g); i <= group_end(g); ++i) {
          invariant(i >= 0 && i < x.rows(), "GroupPartition::group: index out of bounds");
          indices.push_back(i);
        }
      }

      return x(indices, Eigen::all);
    }

    /**
       * @brief Row indices of a group (or supergroup) as an owned Eigen vector.
       *
       * For callers that need a zero-copy `x(idx, all)` view participating in a
       * product: keep the returned vector alive in scope and index with it. An
       * Eigen index type (rather than the `std::vector<int>` used by `group()`)
       * avoids a spurious GCC-14 `-Warray-bounds` on the index copy inside
       * Eigen's product evaluator (fixed in GCC 15).
       */
    Eigen::VectorXi group_indices(Group const& group) const {
      std::vector<int> indices;

      auto const& subs = this->subgroups.at(group);

      for (auto const& g : subs) {
        for (int i = group_start(g); i <= group_end(g); ++i) {
          indices.push_back(i);
        }
      }

      return Eigen::VectorXi::Map(indices.data(), static_cast<Eigen::Index>(indices.size()));
    }

    /**
       * @brief Extract all rows across all groups.
       *
       * @param x  Feature matrix (n × p).
       * @return   Sub-matrix with all grouped rows.
       */
    template<typename Derived> auto data(Eigen::MatrixBase<Derived> const& x) const {
      std::vector<int> indices;

      for (auto const& kv : blocks) {
        auto const& g = kv.first;
        for (int i = group_start(g); i <= group_end(g); ++i) {
          indices.push_back(i);
        }
      }

      return x(indices, Eigen::all);
    }

    /** @brief Overall mean of all grouped rows (p). */
    types::FeatureVector mean(types::FeatureMatrix const& x) const;
    /** @brief Between-group sum of squares matrix (p × p). */
    types::FeatureMatrix bgss(types::FeatureMatrix const& x) const;
    /** @brief Within-group sum of squares matrix (p × p). */
    types::FeatureMatrix wgss(types::FeatureMatrix const& x) const;

    /**
       * @brief Create a partition containing only the given groups.
       *
       * @param groups  Set of group labels to keep.
       * @return        New GroupPartition restricted to @p groups.
       */
    GroupPartition subset(GroupSet const& groups) const;

    using SplitSizes = std::map<types::GroupId, int>;

    /**
       * @brief Split each group's block into left and right children.
       *
       * For each leaf group, @p left_sizes specifies how many rows go to
       * the left child (the first rows of the block).  The remaining rows
       * go to the right child.  Groups absent from @p left_sizes go
       * entirely to the right child (left_count = 0).
       *
       * The caller is responsible for having already reordered rows within
       * each block so that left-bound observations come first.
       *
       * @param left_sizes  Maps each leaf group to its left child row count.
       * @return            Pair of {left, right} GroupPartitions.
       */
    std::pair<GroupPartition, GroupPartition> split(SplitSizes const& left_sizes) const;

    /**
       * @brief Merge groups according to a mapping.
       *
       * @param mapping  Maps original group labels to supergroup labels.
       * @return         New GroupPartition with merged groups.
       */
    GroupPartition remap(GroupMap const& mapping) const;

    /**
       * @brief Collapse all groups into a single supergroup.
       *
       * @return  New GroupPartition with one supergroup containing all groups.
       */
    GroupPartition collapse() const;

    /** @brief Set of all group labels in this partition. */
    GroupSet const groups;
    /** @brief Maps each group to its supergroup (identity if no merge). */
    GroupMap const supergroups;
    /** @brief Maps each group to its set of subgroups. */
    GroupInvMap const subgroups;

  private:
    struct Block {
      int start                          = 0;
      int end                            = 0;
      int size                           = 0;
      std::optional<types::GroupId> next = std::nullopt;
      std::optional<types::GroupId> prev = std::nullopt;
    };

    using BlockMap = std::map<types::GroupId, Block>;
    BlockMap const blocks;

    static BlockMap init_blocks(GroupVector const& y);
    static GroupMap init_supergroups(GroupSet const& groups);

    explicit GroupPartition(BlockMap const& blocks);

    GroupPartition(BlockMap const& blocks, GroupSet const& groups);

    GroupPartition(BlockMap const& blocks, GroupMap const& supergroups);

    GroupPartition(BlockMap const& blocks, GroupSet const& groups, GroupMap const& supergroups);
  };
}
