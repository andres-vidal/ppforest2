#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "models/strategies/stop/CompositeStop.hpp"
#include "models/strategies/stop/StopRule.hpp"
#include "test/MockStop.hpp"
#include "test/NodeContextFixture.hpp"
#include "utils/Types.hpp"
#include "utils/Macros.hpp"

#include <memory>

using namespace ppforest2;
using namespace ppforest2::stop;
using namespace ppforest2::stats;
using namespace ppforest2::test;
using namespace ppforest2::types;
using json = nlohmann::json;

TEST(CompositeStop, FromJsonRoundTrip) {
  json const j = {
      {"name", "any"}, {"rules", json::array({{{"name", "pure_node"}}, {{"name", "min_size"}, {"min_size", 5}}})}
  };

  auto strategy = CompositeStop::from_json(j);

  EXPECT_EQ(strategy->to_json(), j);
}

TEST(CompositeStop, FromJsonUnknownParam) {
  json const j = {{"name", "any"}, {"rules", json::array({{{"name", "pure_node"}}})}, {"extra", 1}};
  EXPECT_THROW(CompositeStop::from_json(j), std::runtime_error);
}

TEST(CompositeStop, RegistryLookup) {
  json const j = {
      {"name", "any"}, {"rules", json::array({{{"name", "pure_node"}}, {{"name", "min_size"}, {"min_size", 5}}})}
  };

  auto strategy = StopRule::from_json(j);
  ASSERT_NE(strategy, nullptr);

  auto out = strategy->to_json();
  EXPECT_EQ(out, j);
}

TEST(CompositeStop, DisplayName) {
  auto rule = any({pure_node(), min_size(5)});
  EXPECT_EQ(rule->display_name(), "Any(Pure node, Min size (5))");
}

TEST(CompositeStop, FiresAndShortCircuitsWhenFirstRuleFires) {
  NodeContextFixture f(MAT(Feature, rows(2), 1, 0, 0, 1), VEC(GroupId, 0, 1));
  auto first  = std::make_shared<MockStop>(true);
  auto second = std::make_shared<MockStop>(false);

  auto rule = any({first, second});

  EXPECT_TRUE(rule->should_stop(f.ctx, f.rng));
  EXPECT_EQ(first->compute_calls, 1);
  EXPECT_EQ(second->compute_calls, 0) << "any() short-circuits after the first true verdict";
}

TEST(CompositeStop, FiresWhenLaterRuleFires) {
  NodeContextFixture f(MAT(Feature, rows(2), 1, 0, 0, 1), VEC(GroupId, 0, 1));
  auto first  = std::make_shared<MockStop>(false);
  auto second = std::make_shared<MockStop>(true);
  auto rule   = any({first, second});

  EXPECT_TRUE(rule->should_stop(f.ctx, f.rng));
  EXPECT_EQ(first->compute_calls, 1);
  EXPECT_EQ(second->compute_calls, 1) << "second rule consulted because first returned false";
}

TEST(CompositeStop, DoesNotFireWhenNoRuleFires) {
  NodeContextFixture f(MAT(Feature, rows(2), 1, 0, 0, 1), VEC(GroupId, 0, 1));
  auto first  = std::make_shared<MockStop>(false);
  auto second = std::make_shared<MockStop>(false);
  auto rule   = any({first, second});

  EXPECT_FALSE(rule->should_stop(f.ctx, f.rng));
  EXPECT_EQ(first->compute_calls, 1);
  EXPECT_EQ(second->compute_calls, 1);
}

TEST(CompositeStop, RejectsEmptyRules) {
  // An empty composite is a no-op: `should_stop` would never fire and
  // `supported_modes` would claim both modes. Combined with a grouping
  // strategy that can fail to make progress (e.g. ByCutpoint), this yields
  // unbounded recursion. Reject at construction time.
  EXPECT_THROW(any({}), std::exception);
}

TEST(CompositeStop, FromJsonRejectsEmptyRules) {
  nlohmann::json const j = {{"name", "any"}, {"rules", nlohmann::json::array()}};
  EXPECT_THROW(CompositeStop::from_json(j), std::exception);
}
