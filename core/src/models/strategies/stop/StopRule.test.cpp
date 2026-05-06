/**
 * @file StopRule.test.cpp
 * @brief Tests for the `StopRule` base-class NVI: the tree-level
 *        invariant that short-circuits stopping when the partition has
 *        fewer than 2 groups, and the delegation to the subclass
 *        `compute` otherwise. Concrete subclasses (`PureNode`,
 *        `MinSize`, ...) are exercised elsewhere.
 */
#include <gtest/gtest.h>

#include "models/strategies/stop/StopRule.hpp"
#include "models/strategies/NodeContext.hpp"
#include "test/MockStop.hpp"
#include "test/NodeContextFixture.hpp"
#include "stats/Stats.hpp"
#include "utils/Macros.hpp"
#include "utils/Types.hpp"

using namespace ppforest2;
using namespace ppforest2::stop;
using namespace ppforest2::stats;
using namespace ppforest2::test;
using namespace ppforest2::types;

TEST(StopRuleNVI, StopsOnSingleGroupWithoutCallingCompute) {
  // Tree-level invariant: 1-group partition is unsplittable, so stop
  // regardless of the configured rule.
  NodeContextFixture f(MAT(Feature, rows(2), 1, 0, 0, 1), VEC(GroupId, 0, 0));

  // would say "don't stop" if asked
  MockStop strategy(false);

  EXPECT_TRUE(strategy.should_stop(f.ctx, f.rng));
  EXPECT_EQ(strategy.compute_calls, 0);
}

TEST(StopRuleNVI, DelegatesToComputeOnTwoOrMoreGroups) {
  NodeContextFixture f(MAT(Feature, rows(2), 1, 0, 0, 1), VEC(GroupId, 0, 1));

  MockStop stop_true(true);
  EXPECT_TRUE(stop_true.should_stop(f.ctx, f.rng));
  EXPECT_EQ(stop_true.compute_calls, 1);

  MockStop stop_false(false);
  EXPECT_FALSE(stop_false.should_stop(f.ctx, f.rng));
  EXPECT_EQ(stop_false.compute_calls, 1);
}
