#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "models/strategies/grouping/ByLabel.hpp"
#include "models/strategies/grouping/Grouping.hpp"
#include "models/strategies/NodeContext.hpp"
#include "test/NodeContextFixture.hpp"
#include "utils/Types.hpp"
#include "utils/Macros.hpp"

using namespace ppforest2;
using namespace ppforest2::grouping;
using namespace ppforest2::stats;
using namespace ppforest2::test;
using namespace ppforest2::types;
using json = nlohmann::json;


TEST(ByLabelGrouping, FromJsonValid) {
  json const j = {{"name", "by_label"}};

  auto strategy = ByLabel::from_json(j);

  ASSERT_NE(strategy, nullptr);
}

TEST(ByLabelGrouping, FromJsonRoundTrip) {
  json const j = {{"name", "by_label"}};

  auto strategy = ByLabel::from_json(j);

  EXPECT_EQ(strategy->to_json(), j);
}

TEST(ByLabelGrouping, FromJsonUnknownParam) {
  json const j = {{"name", "by_label"}, {"extra", 0}};
  EXPECT_THROW(ByLabel::from_json(j), std::runtime_error);
}

TEST(ByLabelGrouping, RegistryLookup) {
  json const j = {{"name", "by_label"}};

  auto strategy = Grouping::from_json(j);

  ASSERT_NE(strategy, nullptr);
  EXPECT_EQ(strategy->to_json(), j);
}

TEST(ByLabelGrouping, RegistryUnknownStrategy) {
  json const j = {{"name", "unknown_grouping"}};
  EXPECT_THROW(Grouping::from_json(j), std::runtime_error);
}

TEST(ByLabelGrouping, InitCreatesGroupPartition) {
  GroupIdVector const y = VEC(GroupId, 0, 0, 1, 1, 2, 2);

  auto y_part = ByLabel().init(y);

  EXPECT_EQ(y_part.groups, (std::set<int>{0, 1, 2}));
  EXPECT_EQ(y_part.group_size(0) + y_part.group_size(1) + y_part.group_size(2), 6);
}

TEST(ByLabelGrouping, SplitsBinaryPartition) {
  // 3 original groups remapped to 2 binary groups: {0,1} -> 0, {2} -> 1
  FeatureMatrix const x = MAT(Feature, rows(6), 1, 2, 3, 4, 5, 6);
  GroupIdVector const y = VEC(GroupId, 0, 0, 1, 1, 2, 2);

  NodeContextFixture f(x, y);
  f.ctx.y_bin.emplace(f.ctx.y.remap({{0, 0}, {1, 0}, {2, 1}}));
  f.ctx.lower_group = 0;
  f.ctx.upper_group = 1;

  ByLabel().split(f.ctx, f.rng);

  EXPECT_EQ(f.ctx.lower_y_part->groups, (std::set<int>{0, 1}));
  EXPECT_EQ(f.ctx.upper_y_part->groups, (std::set<int>{2}));
}

TEST(ByLabelGrouping, AllRowsAccountedFor) {
  // 4 original groups remapped: {0,1} -> 0, {2,3} -> 1
  FeatureMatrix const x = MAT(Feature, rows(8), 1, 2, 3, 4, 5, 6, 7, 8);
  GroupIdVector const y = VEC(GroupId, 0, 0, 1, 1, 2, 2, 3, 3);

  NodeContextFixture f(x, y);
  f.ctx.y_bin.emplace(f.ctx.y.remap({{0, 0}, {1, 0}, {2, 1}, {3, 1}}));
  f.ctx.lower_group = 0;
  f.ctx.upper_group = 1;
  ByLabel().split(f.ctx, f.rng);

  auto lower_data = f.ctx.lower_y_part->data(x).eval();
  auto upper_data = f.ctx.upper_y_part->data(x).eval();

  EXPECT_EQ_DATA(lower_data, (MAT(Feature, rows(4), 1, 2, 3, 4)));
  EXPECT_EQ_DATA(upper_data, (MAT(Feature, rows(4), 5, 6, 7, 8)));
}

TEST(ByLabelGrouping, AllGroupsAccountedFor) {
  FeatureMatrix const x = MAT(Feature, rows(8), 1, 2, 3, 4, 5, 6, 7, 8);
  GroupIdVector const y = VEC(GroupId, 0, 0, 1, 1, 2, 2, 3, 3);

  NodeContextFixture f(x, y);
  f.ctx.y_bin.emplace(f.ctx.y.remap({{0, 0}, {1, 0}, {2, 1}, {3, 1}}));
  f.ctx.lower_group = 0;
  f.ctx.upper_group = 1;

  ByLabel().split(f.ctx, f.rng);

  auto const& lower = f.ctx.lower_y_part.value();
  auto const& upper = f.ctx.upper_y_part.value();


  std::set<int> group_union;
  std::set_union(
      lower.groups.begin(),
      lower.groups.end(),
      upper.groups.begin(),
      upper.groups.end(),
      std::inserter(group_union, group_union.begin())
  );

  EXPECT_EQ(group_union, f.ctx.y.groups);
}

TEST(ByLabelGrouping, NoOverlap) {
  FeatureMatrix const x = MAT(Feature, rows(6), 1, 2, 3, 4, 5, 6);
  GroupIdVector const y = VEC(GroupId, 0, 0, 1, 1, 2, 2);

  NodeContextFixture f(x, y);
  f.ctx.y_bin.emplace(f.ctx.y.remap({{0, 0}, {1, 1}, {2, 1}}));
  f.ctx.lower_group = 0;
  f.ctx.upper_group = 1;

  ByLabel().split(f.ctx, f.rng);

  auto const& lower = f.ctx.lower_y_part.value();
  auto const& upper = f.ctx.upper_y_part.value();

  std::set<int> group_intersection;
  std::set_intersection(
      lower.groups.begin(),
      lower.groups.end(),
      upper.groups.begin(),
      upper.groups.end(),
      std::inserter(group_intersection, group_intersection.begin())
  );

  EXPECT_EQ(group_intersection, std::set<int>()) << "Groups in lower and upper should not overlap";
}

TEST(ByLabelGrouping, NodeContextInterface) {
  // Test the NodeContext-based split() method
  FeatureMatrix const x = MAT(Feature, rows(6), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
  GroupIdVector const y = VEC(GroupId, 0, 0, 1, 1, 2, 2);

  NodeContextFixture f(x, y);

  f.ctx.y_bin.emplace(GroupPartition(y).remap({{0, 0}, {1, 0}, {2, 1}}));
  f.ctx.lower_group = 0;
  f.ctx.upper_group = 1;

  ByLabel().split(f.ctx, f.rng);

  ASSERT_TRUE(f.ctx.lower_y_part.has_value());
  ASSERT_TRUE(f.ctx.upper_y_part.has_value());
  EXPECT_EQ(f.ctx.lower_y_part->groups, (std::set<int>{0, 1}));
  EXPECT_EQ(f.ctx.upper_y_part->groups, (std::set<int>{2}));
}

TEST(ByLabelGrouping, DisplayName) {
  EXPECT_EQ(ByLabel().display_name(), "By label");
}
