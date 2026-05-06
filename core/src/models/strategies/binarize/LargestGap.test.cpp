#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "models/strategies/binarize/LargestGap.hpp"
#include "models/strategies/binarize/Binarization.hpp"
#include "models/strategies/NodeContext.hpp"
#include "test/NodeContextFixture.hpp"
#include "utils/Types.hpp"
#include "utils/Macros.hpp"

using namespace ppforest2;
using namespace ppforest2::binarize;
using namespace ppforest2::stats;
using namespace ppforest2::test;
using namespace ppforest2::types;
using json = nlohmann::json;


TEST(LargestGapBinarize, FromJsonValid) {
  json const j  = {{"name", "largest_gap"}};
  auto strategy = LargestGap::from_json(j);
  ASSERT_NE(strategy, nullptr);
}

TEST(LargestGapBinarize, FromJsonRoundTrip) {
  json const j = {{"name", "largest_gap"}};

  auto strategy = LargestGap::from_json(j);

  EXPECT_EQ(strategy->to_json(), j);
}

TEST(LargestGapBinarize, FromJsonUnknownParam) {
  json const j = {{"name", "largest_gap"}, {"extra", true}};
  EXPECT_THROW(LargestGap::from_json(j), std::runtime_error);
}

TEST(LargestGapBinarize, RegistryLookup) {
  json const j = {{"name", "largest_gap"}};

  auto strategy = Binarization::from_json(j);

  ASSERT_NE(strategy, nullptr);
  EXPECT_EQ(strategy->to_json(), j);
}

TEST(LargestGapBinarize, RegistryUnknownStrategy) {
  json const j = {{"name", "unknown_binarize"}};
  EXPECT_THROW(Binarization::from_json(j), std::runtime_error);
}


TEST(LargestGapBinarize, ThreeGroupsSplitByLargestGap) {
  // 2-D feature matrix with the discriminating signal in column 0 and a
  // constant noise column. Under projector [1, 0] the projected means
  // are 1.0, 2.0, 10.0 — largest gap is between 2.0 and 10.0, so binary
  // group 0 = {0, 1} and binary group 1 = {2}.
  FeatureMatrix const x = MAT(Feature, rows(6), 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 2.0, 0.0, 10.0, 0.0, 10.0, 0.0);
  GroupIdVector const y = VEC(GroupId, 0, 0, 1, 1, 2, 2);

  NodeContextFixture f(x, y);
  f.ctx.projector = VEC(Feature, 1, 0);

  LargestGap().regroup(f.ctx, f.rng);

  auto const& result = f.ctx.y_bin.value();

  EXPECT_EQ(result.groups.size(), 2U);

  auto const group_0   = *result.groups.begin();
  auto const group_0_x = result.group(x, group_0).eval();
  auto const group_0_y = result.group(y, group_0).eval();

  EXPECT_EQ(group_0, 0);
  EXPECT_EQ_DATA(group_0_x, (MAT(Feature, rows(4), 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 2.0, 0.0)));
  EXPECT_EQ_DATA(group_0_y, (VEC(GroupId, 0, 0, 1, 1)));

  auto const group_1   = *std::next(result.groups.begin());
  auto const group_1_x = result.group(x, group_1).eval();
  auto const group_1_y = result.group(y, group_1).eval();

  EXPECT_EQ(group_1, 1);
  EXPECT_EQ_DATA(group_1_x, (MAT(Feature, rows(2), 10.0, 0.0, 10.0, 0.0)));
  EXPECT_EQ_DATA(group_1_y, (VEC(GroupId, 2, 2)));
}

TEST(LargestGapBinarize, ThreeGroupsPreservesAllObservations) {
  // Same data shape as above. Verifies that the binarized partition
  // still indexes every original row.
  FeatureMatrix const x = MAT(Feature, rows(6), 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 2.0, 0.0, 10.0, 0.0, 10.0, 0.0);
  GroupIdVector const y = VEC(GroupId, 0, 0, 1, 1, 2, 2);

  NodeContextFixture f(x, y);
  f.ctx.projector = VEC(Feature, 1, 0);

  LargestGap().regroup(f.ctx, f.rng);
  auto const& result = f.ctx.y_bin.value();

  auto const data = result.data(x).eval();
  EXPECT_EQ_DATA(data, x);
}

TEST(LargestGapBinarize, FourGroupsSplitCorrectly) {
  // Four groups projected to means 1, 2, 3, 100 → largest gap is
  // between 3 and 100, so {0, 1, 2} → bin 0 and {3} → bin 1.
  FeatureMatrix const x =
      MAT(Feature, rows(8), 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 2.0, 0.0, 3.0, 0.0, 3.0, 0.0, 100.0, 0.0, 100.0, 0.0);
  GroupIdVector const y = VEC(GroupId, 0, 0, 1, 1, 2, 2, 3, 3);

  NodeContextFixture f(x, y);
  f.ctx.projector = VEC(Feature, 1, 0);

  LargestGap().regroup(f.ctx, f.rng);
  auto const& result = f.ctx.y_bin.value();

  EXPECT_EQ(result.groups.size(), 2U);

  auto const group_0   = *result.groups.begin();
  auto const group_0_x = result.group(x, group_0).eval();
  auto const group_0_y = result.group(y, group_0).eval();

  EXPECT_EQ(group_0, 0);
  EXPECT_EQ_DATA(group_0_x, (MAT(Feature, rows(6), 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 2.0, 0.0, 3.0, 0.0, 3.0, 0.0)));
  EXPECT_EQ_DATA(group_0_y, (VEC(GroupId, 0, 0, 1, 1, 2, 2)));

  auto const group_1   = *std::next(result.groups.begin());
  auto const group_1_x = result.group(x, group_1).eval();
  auto const group_1_y = result.group(y, group_1).eval();

  EXPECT_EQ(group_1, 1);
  EXPECT_EQ_DATA(group_1_x, (MAT(Feature, rows(2), 100.0, 0.0, 100.0, 0.0)));
  EXPECT_EQ_DATA(group_1_y, (VEC(GroupId, 3, 3)));

  auto const data = result.data(x).eval();
  EXPECT_EQ_DATA(data, x);
}

TEST(LargestGapBinarize, EqualMeansDoesNotCrash) {
  // All groups project to the same mean → every gap is 0. LargestGap
  // still has to fall back to a valid binary partition rather than
  // crashing.
  FeatureMatrix const x = MAT(Feature, rows(6), 5.0, 0.0, 5.0, 0.0, 5.0, 0.0, 5.0, 0.0, 5.0, 0.0, 5.0, 0.0);
  GroupIdVector const y = VEC(GroupId, 0, 0, 1, 1, 2, 2);

  NodeContextFixture f(x, y);
  f.ctx.projector = VEC(Feature, 1, 0);

  LargestGap().regroup(f.ctx, f.rng);

  ASSERT_EQ(f.ctx.y_bin.value().groups.size(), 2U);
}

TEST(LargestGapBinarize, NodeContextInterface) {
  // 3 groups, feature space where group 2 is far away on dimension 0.
  FeatureMatrix const x = MAT(Feature, rows(6), 1, 0, 2, 0, 3, 0, 4, 0, 100, 0, 101, 0);
  GroupIdVector const y = VEC(GroupId, 0, 0, 1, 1, 2, 2);

  NodeContextFixture f(x, y);
  f.ctx.projector = VEC(Feature, 1, 0);

  LargestGap const lg;
  lg.regroup(f.ctx, f.rng);

  ASSERT_TRUE(f.ctx.y_bin.has_value());
  EXPECT_EQ(f.ctx.y_bin->groups.size(), 2U); // NOLINT(bugprone-unchecked-optional-access)
}

TEST(LargestGapBinarize, DisplayName) {
  LargestGap const lg;
  EXPECT_EQ(lg.display_name(), "Largest gap");
}
