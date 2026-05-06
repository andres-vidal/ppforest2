#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "models/strategies/stop/MinVariance.hpp"
#include "models/strategies/stop/StopRule.hpp"
#include "test/NodeContextFixture.hpp"
#include "utils/Types.hpp"
#include "utils/Macros.hpp"

using namespace ppforest2;
using namespace ppforest2::stop;
using namespace ppforest2::stats;
using namespace ppforest2::test;
using namespace ppforest2::types;
using json = nlohmann::json;

TEST(MinVarianceStop, FromJsonValid) {
  json const j  = {{"name", "min_variance"}, {"threshold", 0.01F}};
  auto strategy = MinVariance::from_json(j);
  ASSERT_NE(strategy, nullptr);
}

TEST(MinVarianceStop, FromJsonRoundTrip) {
  json const j  = {{"name", "min_variance"}, {"threshold", 0.01F}};
  auto strategy = MinVariance::from_json(j);

  auto out = strategy->to_json();

  EXPECT_EQ(out, j);
}

TEST(MinVarianceStop, FromJsonUnknownParam) {
  json const j = {{"name", "min_variance"}, {"threshold", 0.01F}, {"extra", 1}};
  EXPECT_THROW(MinVariance::from_json(j), std::runtime_error);
}

TEST(MinVarianceStop, RegistryLookup) {
  json const j  = {{"name", "min_variance"}, {"threshold", 0.01F}};
  auto strategy = StopRule::from_json(j);
  ASSERT_NE(strategy, nullptr);

  auto out = strategy->to_json();
  EXPECT_EQ(out, j);
}

TEST(MinVarianceStop, DisplayName) {
  // Default-float formatting renders small thresholds compactly and falls
  // back to scientific when fixed-notation would round them to zero.
  EXPECT_EQ(MinVariance(0.01F).display_name(), "Min variance (0.01)");
  EXPECT_EQ(MinVariance(1e-6F).display_name(), "Min variance (1e-06)");
}

TEST(MinVarianceStop, StopsWhenVarianceIsZero) {
  // Regression-flavour fixture builds the 2-group median-split partition
  // ByCutpoint would have set up at the root; single-group partitions
  // hit the tree-level guard in `StopRule::should_stop` instead.
  NodeContextFixture f(MAT(Feature, rows(3), 1, 2, 3, 4, 5, 6), VEC(Outcome, 1.0, 1.0, 1.0));

  MinVariance const rule(0.01F);
  EXPECT_TRUE(rule.should_stop(f.ctx, f.rng));
}

TEST(MinVarianceStop, DoesNotStopWhenVarianceIsHigh) {
  NodeContextFixture f(MAT(Feature, rows(3), 1, 2, 3, 4, 5, 6), VEC(Outcome, 1.0, 10.0, 20.0));

  MinVariance const rule(0.01F);
  EXPECT_FALSE(rule.should_stop(f.ctx, f.rng));
}

TEST(MinVarianceStop, StopsWithSingleObservation) {
  // 2 rows, 2 single-row groups. `MinVariance::compute` treats `n <= 1`
  // (size of the variance sample) as a stop, and here `ctx.y.data(...)`
  // returns both rows — so this exercises the variance-zero path rather
  // than the n=1 branch. Kept as documentation of tree-level behavior:
  // small leaves with constant response should stop.
  NodeContextFixture f(MAT(Feature, rows(2), 1, 2, 3, 4), VEC(Outcome, 5.0, 5.0));

  MinVariance const rule(0.01F);
  EXPECT_TRUE(rule.should_stop(f.ctx, f.rng));
}
