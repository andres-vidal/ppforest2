#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "models/strategies/stop/MaxDepth.hpp"
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

TEST(MaxDepthStop, FromJsonValid) {
  json const j  = {{"name", "max_depth"}, {"max_depth", 3}};
  auto strategy = MaxDepth::from_json(j);
  ASSERT_NE(strategy, nullptr);
}

TEST(MaxDepthStop, FromJsonRoundTrip) {
  json const j  = {{"name", "max_depth"}, {"max_depth", 3}};
  auto strategy = MaxDepth::from_json(j);
  EXPECT_EQ(strategy->to_json(), j);
}

TEST(MaxDepthStop, FromJsonUnknownParam) {
  json const j = {{"name", "max_depth"}, {"max_depth", 3}, {"extra", 1}};
  EXPECT_THROW(MaxDepth::from_json(j), std::runtime_error);
}

TEST(MaxDepthStop, RegistryLookup) {
  json const j  = {{"name", "max_depth"}, {"max_depth", 3}};
  auto strategy = StopRule::from_json(j);
  ASSERT_NE(strategy, nullptr);
  EXPECT_EQ(strategy->to_json(), j);
}

TEST(MaxDepthStop, DisplayName) {
  MaxDepth const rule(3);
  EXPECT_EQ(rule.display_name(), "Max depth (3)");
}

TEST(MaxDepthStop, StopsWhenDepthReachesMax) {
  // Rule is "stop when ctx.depth >= max_depth" — fires at the boundary.
  MaxDepth const rule(3);

  NodeContextFixture f3(MAT(Feature, rows(2), 1, 2, 3, 4), VEC(GroupId, 0, 1));
  f3.ctx.depth = 3;
  EXPECT_TRUE(rule.should_stop(f3.ctx, f3.rng));

  NodeContextFixture f4(MAT(Feature, rows(2), 1, 2, 3, 4), VEC(GroupId, 0, 1));
  f4.ctx.depth = 4;
  EXPECT_TRUE(rule.should_stop(f4.ctx, f4.rng));

  NodeContextFixture f2(MAT(Feature, rows(2), 1, 2, 3, 4), VEC(GroupId, 0, 1));
  f2.ctx.depth = 2;
  EXPECT_FALSE(rule.should_stop(f2.ctx, f2.rng));
}

TEST(MaxDepthStop, ZeroMaxDepthProducesStump) {
  // max_depth=0 fires at the root — unusual but explicitly allowed.
  NodeContextFixture f(MAT(Feature, rows(2), 1, 2, 3, 4), VEC(GroupId, 0, 1));
  // f.ctx.depth defaults to 0

  MaxDepth const rule(0);
  EXPECT_TRUE(rule.should_stop(f.ctx, f.rng));
}

TEST(MaxDepthStop, RejectsNegativeMaxDepth) {
  // A negative max would fire at the root for every tree. The constructor
  // rejects it to prevent silently-degenerate models.
  EXPECT_THROW(MaxDepth(-1), std::exception);
  EXPECT_THROW(MaxDepth(-100), std::exception);
}
