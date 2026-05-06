#include "models/strategies/grouping/ByCutpoint.hpp"

#include "models/strategies/NodeContext.hpp"
#include "test/NodeContextFixture.hpp"
#include "utils/Macros.hpp"

#include <gtest/gtest.h>

using namespace ppforest2;
using namespace ppforest2::grouping;
using namespace ppforest2::stats;
using namespace ppforest2::test;
using namespace ppforest2::types;

TEST(ByCutpoint, FromJsonValid) {
  nlohmann::json const j = {{"name", "by_cutpoint"}};

  auto strategy = ByCutpoint::from_json(j);

  ASSERT_NE(strategy, nullptr);
}

TEST(ByCutpoint, FromJsonRoundTrip) {
  nlohmann::json const j = {{"name", "by_cutpoint"}};

  auto strategy = ByCutpoint::from_json(j);

  EXPECT_EQ(strategy->to_json(), j);
}

TEST(ByCutpoint, FromJsonUnknownParam) {
  nlohmann::json const j = {{"name", "by_cutpoint"}, {"extra", 0}};
  EXPECT_THROW(ByCutpoint::from_json(j), std::runtime_error);
}

TEST(ByCutpoint, RegistryLookup) {
  nlohmann::json const j = {{"name", "by_cutpoint"}};

  auto strategy = Grouping::from_json(j);

  ASSERT_NE(strategy, nullptr);
  EXPECT_EQ(strategy->to_json(), j);
}

TEST(ByCutpoint, RegistryUnknownStrategy) {
  nlohmann::json const j = {{"name", "unknown_grouping"}};
  EXPECT_THROW(Grouping::from_json(j), std::runtime_error);
}

TEST(ByCutpoint, FactoryFunction) {
  auto ptr = by_cutpoint();
  EXPECT_NE(ptr, nullptr);
  EXPECT_EQ(ptr->display_name(), "By cutpoint");
}

TEST(ByCutpoint, DisplayName) {
  ByCutpoint strategy;
  EXPECT_EQ(strategy.display_name(), "By cutpoint");
}

TEST(ByCutpoint, InitCreatesGroupPartition) {
  GroupIdVector y(6);
  y << 0, 0, 0, 1, 1, 1;

  ByCutpoint strategy;
  GroupPartition gp = strategy.init(y);

  EXPECT_EQ(gp.groups.size(), 2);
  EXPECT_EQ(gp.group_size(0), 3);
  EXPECT_EQ(gp.group_size(1), 3);
}

TEST(ByCutpoint, InitHandlesSingleObservation) {
  GroupIdVector y(1);
  y << 0;

  ByCutpoint strategy;
  GroupPartition gp = strategy.init(y);

  EXPECT_EQ(gp.groups.size(), 1);
  EXPECT_EQ(gp.group_size(0), 1);
}

TEST(ByCutpoint, InitMedianSplitsOddSizes) {
  // n=5: mid=5/2=2 → group 0 covers [0, 1], group 1 covers [2, 4].
  GroupIdVector y(5);
  y << 0, 0, 1, 1, 1;

  ByCutpoint strategy;
  GroupPartition gp = strategy.init(y);

  EXPECT_EQ(gp.groups.size(), 2);
  EXPECT_EQ(gp.group_size(0), 2);
  EXPECT_EQ(gp.group_size(1), 3);
}

TEST(ByCutpoint, SupportsRegressionOnly) {
  ByCutpoint strategy;
  auto const modes = strategy.supported_modes();
  EXPECT_EQ(modes.count(types::Mode::Regression), 1U);
  EXPECT_EQ(modes.count(types::Mode::Classification), 0U);
}

// ---------------------------------------------------------------------------
// split() — main work
// ---------------------------------------------------------------------------

namespace {
  // Set the fields `ByCutpoint::split` expects: projector, cutpoint, and
  // the lower/upper group labels that the cutpoint NVI's orient step
  // would have written.
  void
  configure(NodeContext& ctx, std::vector<Feature> const& projector, Feature cutpoint, GroupId lower, GroupId upper) {
    Projector p(projector.size());
    for (std::size_t i = 0; i < projector.size(); ++i) {
      p(static_cast<int>(i)) = projector[i];
    }
    ctx.projector   = p;
    ctx.cutpoint    = cutpoint;
    ctx.lower_group = lower;
    ctx.upper_group = upper;
  }
}

TEST(ByCutpoint, SplitPartitionsRowsByCutpoint) {
  // Projector [1, 0], cutpoint 3.0. Rows with x[0] < 3 → lower child, else upper.
  // Initial 2-group layout: group 0 = rows [0,1,2], group 1 = rows [3,4,5].
  NodeContextFixture f(
      MAT(Feature, rows(6), 1.0, 0.0, 4.0, 0.0, 2.0, 0.0, 5.0, 0.0, 0.5, 0.0, 6.0, 0.0),
      VEC(Outcome, 1.0, 4.0, 2.0, 5.0, 0.5, 6.0)
  );
  configure(f.ctx, {1.0F, 0.0F}, 3.0F, 0, 1);

  ByCutpoint const strategy;
  strategy.split(f.ctx, f.rng);

  ASSERT_TRUE(f.ctx.lower_y_part.has_value());
  ASSERT_TRUE(f.ctx.upper_y_part.has_value());
  // 3 rows have x[0] < 3 (values 1, 2, 0.5) → lower; 3 have x[0] ≥ 3 → upper.
  EXPECT_EQ(f.ctx.lower_y_part->total_size(), 3); // NOLINT(bugprone-unchecked-optional-access)
  EXPECT_EQ(f.ctx.upper_y_part->total_size(), 3); // NOLINT(bugprone-unchecked-optional-access)
}

TEST(ByCutpoint, SplitReordersRowsAndPreservesYOrderInPlace) {
  // ByCutpoint's contract is that rows are ascending in y_vec on entry.
  // After split, rows with projected value < cutpoint come first, then the
  // rest — and within each child range, rows remain ascending in y_vec
  // because `partition_by_cutpoint` uses `std::stable_partition` (no
  // follow-up sort needed).
  //
  // Layout: rows 0..3 have y = (10, 20, 30, 40) (sorted ascending) and
  // projected values (4, 1, 5, 2). Cutpoint 3 splits as:
  //   - lower (proj < 3): original rows 1, 3 with y = (20, 40)
  //   - upper (proj ≥ 3): original rows 0, 2 with y = (10, 30)
  NodeContextFixture f(
      MAT(Feature, rows(4), 4.0, 0.0, 1.0, 0.0, 5.0, 0.0, 2.0, 0.0), VEC(Outcome, 10.0, 20.0, 30.0, 40.0)
  );
  configure(f.ctx, {1.0F, 0.0F}, 3.0F, 0, 1);

  ByCutpoint const strategy;
  strategy.split(f.ctx, f.rng);

  // Lower child range [0, 1]: ascending in y_vec.
  EXPECT_EQ(f.y_vec(0), 20.0F);
  EXPECT_EQ(f.y_vec(1), 40.0F);
  // Upper child range [2, 3]: ascending in y_vec.
  EXPECT_EQ(f.y_vec(2), 10.0F);
  EXPECT_EQ(f.y_vec(3), 30.0F);

  // x rows follow the same permutation as y_vec.
  EXPECT_EQ(f.x(0, 0), 1.0F);
  EXPECT_EQ(f.x(1, 0), 2.0F);
  EXPECT_EQ(f.x(2, 0), 4.0F);
  EXPECT_EQ(f.x(3, 0), 5.0F);
}

TEST(ByCutpoint, SplitAllRowsLeftWhenCutpointIsHigh) {
  // Cutpoint so high that every projected value falls below it → left_n == 0 branch.
  NodeContextFixture f(
      MAT(Feature, rows(4), 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0), VEC(Outcome, 10.0, 20.0, 30.0, 40.0)
  );
  configure(f.ctx, {1.0F, 0.0F}, -1.0F, 0, 1);

  ByCutpoint const strategy;
  strategy.split(f.ctx, f.rng);

  ASSERT_TRUE(f.ctx.lower_y_part.has_value());
  ASSERT_TRUE(f.ctx.upper_y_part.has_value());
  // Both children share the same edge partition (all rows on one side).
  EXPECT_EQ(
      f.ctx.lower_y_part->total_size(), f.ctx.upper_y_part->total_size()
  );                                              // NOLINT(bugprone-unchecked-optional-access)
  EXPECT_EQ(f.ctx.lower_y_part->total_size(), 4); // NOLINT(bugprone-unchecked-optional-access)
}

TEST(ByCutpoint, SplitAllRowsRightWhenCutpointIsLow) {
  // Cutpoint below every projected value → right_n == 0 branch.
  NodeContextFixture f(
      MAT(Feature, rows(4), 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0), VEC(Outcome, 10.0, 20.0, 30.0, 40.0)
  );
  configure(f.ctx, {1.0F, 0.0F}, 100.0F, 0, 1);

  ByCutpoint const strategy;
  strategy.split(f.ctx, f.rng);

  ASSERT_TRUE(f.ctx.lower_y_part.has_value());
  ASSERT_TRUE(f.ctx.upper_y_part.has_value());
  EXPECT_EQ(
      f.ctx.lower_y_part->total_size(), f.ctx.upper_y_part->total_size()
  );                                              // NOLINT(bugprone-unchecked-optional-access)
  EXPECT_EQ(f.ctx.lower_y_part->total_size(), 4); // NOLINT(bugprone-unchecked-optional-access)
}

TEST(ByCutpoint, SplitProducesSingleGroupForSmallChild) {
  // Cutpoint isolates exactly one row in the left child → left_n == 1,
  // which cannot be median-split, so the child is a single-group partition.
  NodeContextFixture f(
      MAT(Feature, rows(4), 1.0, 0.0, 5.0, 0.0, 6.0, 0.0, 7.0, 0.0), VEC(Outcome, 10.0, 50.0, 60.0, 70.0)
  );
  configure(f.ctx, {1.0F, 0.0F}, 2.0F, 0, 1);

  ByCutpoint const strategy;
  strategy.split(f.ctx, f.rng);

  ASSERT_TRUE(f.ctx.lower_y_part.has_value());
  ASSERT_TRUE(f.ctx.upper_y_part.has_value());
  // Lower child has just 1 row → single_group (1 group). Upper has 3 rows → median_split (2 groups).
  EXPECT_EQ(f.ctx.lower_y_part->groups.size(), 1U); // NOLINT(bugprone-unchecked-optional-access)
  EXPECT_EQ(f.ctx.upper_y_part->groups.size(), 2U); // NOLINT(bugprone-unchecked-optional-access)
}
