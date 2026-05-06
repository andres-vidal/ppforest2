#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "models/strategies/stop/PureNode.hpp"
#include "models/strategies/stop/StopRule.hpp"
#include "models/strategies/NodeContext.hpp"
#include "test/NodeContextFixture.hpp"
#include "stats/Stats.hpp"
#include "utils/Types.hpp"
#include "utils/Macros.hpp"

using namespace ppforest2;
using namespace ppforest2::stop;
using namespace ppforest2::stats;
using namespace ppforest2::test;
using namespace ppforest2::types;
using json = nlohmann::json;

TEST(PureNodeStop, FromJsonValid) {
  json const j  = {{"name", "pure_node"}};
  auto strategy = PureNode::from_json(j);
  ASSERT_NE(strategy, nullptr);
}

TEST(PureNodeStop, FromJsonRoundTrip) {
  json const j  = {{"name", "pure_node"}};
  auto strategy = PureNode::from_json(j);

  auto out = strategy->to_json();

  EXPECT_EQ(out, j);
}

TEST(PureNodeStop, FromJsonUnknownParam) {
  json const j = {{"name", "pure_node"}, {"extra", 1}};
  EXPECT_THROW(PureNode::from_json(j), std::runtime_error);
}

TEST(PureNodeStop, RegistryLookup) {
  json const j  = {{"name", "pure_node"}};
  auto strategy = StopRule::from_json(j);
  ASSERT_NE(strategy, nullptr);

  auto out = strategy->to_json();
  EXPECT_EQ(out, j);
}

TEST(PureNodeStop, RegistryUnknownStrategy) {
  json const j = {{"name", "unknown_stop"}};
  EXPECT_THROW(StopRule::from_json(j), std::runtime_error);
}

TEST(PureNodeStop, StopsOnSingleGroup) {
  NodeContextFixture f(MAT(Feature, rows(3), 1, 2, 3, 4, 5, 6), VEC(GroupId, 0, 0, 0));

  PureNode const rule;
  EXPECT_TRUE(rule.should_stop(f.ctx, f.rng));
}

TEST(PureNodeStop, DoesNotStopOnTwoGroups) {
  NodeContextFixture f(MAT(Feature, rows(4), 1, 2, 3, 4, 5, 6, 7, 8), VEC(GroupId, 0, 0, 1, 1));

  PureNode const rule;
  EXPECT_FALSE(rule.should_stop(f.ctx, f.rng));
}

TEST(PureNodeStop, DoesNotStopOnThreeGroups) {
  NodeContextFixture f(MAT(Feature, rows(6), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), VEC(GroupId, 0, 0, 1, 1, 2, 2));

  PureNode const rule;
  EXPECT_FALSE(rule.should_stop(f.ctx, f.rng));
}

TEST(PureNodeStop, IgnoresDepth) {
  // PureNode reads only `ctx.y.groups.size()`, not `ctx.depth`. Build
  // three fixtures at different depths to confirm the verdict doesn't
  // change.
  PureNode const rule;

  NodeContextFixture f0(MAT(Feature, rows(3), 1, 2, 3, 4, 5, 6), VEC(GroupId, 0, 0, 0));
  EXPECT_TRUE(rule.should_stop(f0.ctx, f0.rng));

  NodeContextFixture f10(MAT(Feature, rows(3), 1, 2, 3, 4, 5, 6), VEC(GroupId, 0, 0, 0));
  f10.ctx.depth = 10;
  EXPECT_TRUE(rule.should_stop(f10.ctx, f10.rng));

  NodeContextFixture f100(MAT(Feature, rows(3), 1, 2, 3, 4, 5, 6), VEC(GroupId, 0, 0, 0));
  f100.ctx.depth = 100;
  EXPECT_TRUE(rule.should_stop(f100.ctx, f100.rng));
}

TEST(PureNodeStop, DisplayName) {
  PureNode const rule;
  EXPECT_EQ(rule.display_name(), "Pure node");
}
