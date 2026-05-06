#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "models/strategies/stop/MinSize.hpp"
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

TEST(MinSizeStop, FromJsonValid) {
  json const j  = {{"name", "min_size"}, {"min_size", 5}};
  auto strategy = MinSize::from_json(j);
  ASSERT_NE(strategy, nullptr);
}

TEST(MinSizeStop, FromJsonRoundTrip) {
  json const j  = {{"name", "min_size"}, {"min_size", 5}};
  auto strategy = MinSize::from_json(j);

  auto out = strategy->to_json();

  EXPECT_EQ(out, j);
}

TEST(MinSizeStop, FromJsonUnknownParam) {
  json const j = {{"name", "min_size"}, {"min_size", 5}, {"extra", 1}};
  EXPECT_THROW(MinSize::from_json(j), std::runtime_error);
}

TEST(MinSizeStop, RegistryLookup) {
  json const j  = {{"name", "min_size"}, {"min_size", 5}};
  auto strategy = StopRule::from_json(j);
  ASSERT_NE(strategy, nullptr);

  auto out = strategy->to_json();
  EXPECT_EQ(out, j);
}

TEST(MinSizeStop, DisplayName) {
  MinSize const rule(5);
  EXPECT_EQ(rule.display_name(), "Min size (5)");
}

TEST(MinSizeStop, StopsWhenTooFewObservations) {
  NodeContextFixture f(MAT(Feature, rows(4), 1, 2, 3, 4, 5, 6, 7, 8), VEC(GroupId, 0, 0, 1, 1));

  MinSize const rule(5);
  EXPECT_TRUE(rule.should_stop(f.ctx, f.rng));
}

TEST(MinSizeStop, DoesNotStopWhenEnoughObservations) {
  NodeContextFixture f(MAT(Feature, rows(4), 1, 2, 3, 4, 5, 6, 7, 8), VEC(GroupId, 0, 0, 1, 1));

  MinSize const rule(3);
  EXPECT_FALSE(rule.should_stop(f.ctx, f.rng));
}

TEST(MinSizeStop, ExactBoundaryDoesNotStop) {
  NodeContextFixture f(MAT(Feature, rows(4), 1, 2, 3, 4, 5, 6, 7, 8), VEC(GroupId, 0, 0, 1, 1));

  MinSize const rule(4);
  // total == min_size → does not stop (only stops when total < min_size)
  EXPECT_FALSE(rule.should_stop(f.ctx, f.rng));
}

TEST(MinSizeStop, ExactBoundaryStopsWhenBelow) {
  NodeContextFixture f(MAT(Feature, rows(3), 1, 2, 3, 4, 5, 6), VEC(GroupId, 0, 0, 1));

  MinSize const rule(4);
  // total (3) < min_size (4) -> stops
  EXPECT_TRUE(rule.should_stop(f.ctx, f.rng));
}

TEST(MinSizeStop, RejectsNonPositiveOrUselessThresholds) {
  // 0 and negative thresholds mean the rule never fires; 1 means it only
  // fires on empty nodes. Neither is a meaningful stop condition, so the
  // constructor rejects them.
  EXPECT_THROW(MinSize(0), std::exception);
  EXPECT_THROW(MinSize(-3), std::exception);
  EXPECT_THROW(MinSize(1), std::exception);

  // 2 is the minimum useful value: stop when node has fewer than 2 rows
  // (i.e. 0 or 1, both unsplittable).
  EXPECT_NO_THROW(MinSize(2));
}
