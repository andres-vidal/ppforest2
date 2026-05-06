#include "models/strategies/grouping/ByCutpoint.hpp"

#include "models/strategies/NodeContext.hpp"
#include "utils/RangeVector.hpp"
#include "utils/Invariant.hpp"
#include "utils/JsonReader.hpp"

#include <algorithm>
#include <nlohmann/json.hpp>

using namespace ppforest2::types;
using namespace ppforest2::stats;

namespace ppforest2::grouping {

  namespace {
    /**
     * @brief Partition rows in [start, end] by projected value vs cutpoint.
     *
     * Reorders x and continuous_y in place so that rows with projected
     * value < cutpoint come first. Returns the index of the first right-child
     * row (i.e., left child owns [start, mid-1], right owns [mid, end]).
     *
     * Uses `std::stable_partition`, so the relative order within each side
     * is preserved. When the input range is ascending in `continuous_y`
     * (the contract for ByCutpoint), each child's range is still ascending
     * after this call — no follow-up sort is needed before `median_split`.
     */
    int partition_by_cutpoint(
        FeatureMatrix& x,
        OutcomeVector& continuous_y,
        pp::Projector const& projector,
        Feature cutpoint,
        int start,
        int end
    ) {
      int const n = end - start + 1;

      // 1. Project the node's rows onto the 1-D cutpoint axis.
      FeatureVector const p = x.middleRows(start, n) * projector;

      // 2. Stable index permutation: left-of-cutpoint rows first.
      std::vector<int> perm = utils::range_vector(n);
      auto const pivot      = std::stable_partition(perm.begin(), perm.end(), [&](int i) { return p(i) < cutpoint; });

      // 3. Apply the permutation to x and continuous_y in place.
      x.middleRows(start, n)         = x.middleRows(start, n)(perm, Eigen::all).eval();
      continuous_y.segment(start, n) = continuous_y.segment(start, n)(perm).eval();

      return start + static_cast<int>(std::distance(perm.begin(), pivot));
    }

    /**
   * @brief Create a 2-group partition by median-splitting rows [start, end].
   *
   * Assumes rows are already sorted by continuous_y within this range.
   * Group 0 gets the first half, group 1 gets the second half.
   *
   * Requires n >= 2 (caller must check).
   */
    GroupPartition median_split(int start, int end) {
      int const n   = end - start + 1;
      int const mid = n / 2;

      invariant(n >= 2, "median_split requires at least 2 observations");

      return GroupPartition(start, end).bisect(start + mid);
    }
  }


  nlohmann::json ByCutpoint::to_json() const {
    return {{"name", "by_cutpoint"}};
  }

  GroupPartition ByCutpoint::compute_init(OutcomeVector const& y) const {
    int const n = static_cast<int>(y.size());

    if (n < 2) {
      return GroupPartition(0, n - 1);
    }

    int const mid = n / 2;
    return GroupPartition(0, n - 1).bisect(mid);
  }


  void ByCutpoint::compute(NodeContext& ctx, stats::RNG& /*rng*/) const {
    invariant(ctx.projector.has_value(), "ByCutpoint requires projector on NodeContext");
    invariant(ctx.cutpoint.has_value(), "ByCutpoint requires cutpoint on NodeContext");

    auto lower = ctx.lower_group.value();
    auto upper = ctx.upper_group.value();

    auto const& y_part = ctx.active_partition();

    // Get the contiguous range this node owns.
    int const node_start = std::min(y_part.group_start(lower), y_part.group_start(upper));
    int const node_end   = std::max(y_part.group_end(lower), y_part.group_end(upper));

    // 1. Partition by cutpoint: left-bound rows first.
    int const mid =
        partition_by_cutpoint(ctx.x, ctx.y_vec, ctx.projector.value(), ctx.cutpoint.value(), node_start, node_end);

    int const left_n  = mid - node_start;
    int const right_n = node_end - mid + 1;

    // Edge case: all observations go to one side of the cutpoint. Write
    // identical partitions for both children; the tree builder's no-progress
    // guard (Tree::build_root) detects this and converts the node to a leaf.
    if (left_n == 0) {
      auto edge_y_part = right_n >= 2 ? median_split(mid, node_end) : GroupPartition(mid, node_end);
      ctx.lower_y_part.emplace(edge_y_part);
      ctx.upper_y_part.emplace(edge_y_part);
      return;
    }

    if (right_n == 0) {
      auto edge_y_part = left_n >= 2 ? median_split(node_start, node_end) : GroupPartition(node_start, node_end);
      ctx.lower_y_part.emplace(edge_y_part);
      ctx.upper_y_part.emplace(edge_y_part);
      return;
    }

    // 2. Median-split each child into 2 groups (if enough observations).
    // Children with < 2 observations get a single-group partition;
    // the stop rule will turn them into leaves. `partition_by_cutpoint`
    // uses `std::stable_partition`, so each child's range is still
    // ascending in `continuous_y` (the parent range was, by contract).
    auto make_child_partition = [](int start, int end) -> GroupPartition {
      int const n = end - start + 1;

      if (n < 2) {
        return GroupPartition(start, end);
      }

      return median_split(start, end);
    };

    ctx.lower_y_part.emplace(make_child_partition(node_start, mid - 1));
    ctx.upper_y_part.emplace(make_child_partition(mid, node_end));
  }

  Grouping::Ptr by_cutpoint() {
    return std::make_shared<ByCutpoint>();
  }

  Grouping::Ptr ByCutpoint::from_json(nlohmann::json const& j) {
    JsonReader{j, "by_cutpoint"}.only_keys({"name"});
    return by_cutpoint();
  }
}
