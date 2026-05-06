#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "models/strategies/leaf/MeanResponse.hpp"
#include "models/strategies/leaf/LeafStrategy.hpp"
#include "test/NodeContextFixture.hpp"
#include "utils/Types.hpp"
#include "utils/Macros.hpp"

using namespace ppforest2;
using namespace ppforest2::leaf;
using namespace ppforest2::stats;
using namespace ppforest2::test;
using namespace ppforest2::types;
using json = nlohmann::json;

TEST(MeanResponseLeaf, FromJsonValid) {
  json const j  = {{"name", "mean_response"}};
  auto strategy = MeanResponse::from_json(j);
  ASSERT_NE(strategy, nullptr);
}

TEST(MeanResponseLeaf, FromJsonRoundTrip) {
  json const j = {{"name", "mean_response"}};

  auto strategy = MeanResponse::from_json(j);

  EXPECT_EQ(strategy->to_json(), j);
}

TEST(MeanResponseLeaf, FromJsonUnknownParam) {
  json const j = {{"name", "mean_response"}, {"extra", 0}};
  EXPECT_THROW(MeanResponse::from_json(j), std::runtime_error);
}

TEST(MeanResponseLeaf, RegistryLookup) {
  json const j = {{"name", "mean_response"}};

  auto strategy = LeafStrategy::from_json(j);

  ASSERT_NE(strategy, nullptr);
  EXPECT_EQ(strategy->to_json(), j);
}

TEST(MeanResponseLeaf, DisplayName) {
  MeanResponse const mr;
  EXPECT_EQ(mr.display_name(), "Mean response");
}

TEST(MeanResponseLeaf, MeanOfAllObservations) {
  NodeContextFixture f(MAT(Feature, rows(4), 1, 2, 3, 4, 5, 6, 7, 8), VEC(Outcome, 1.0, 3.0, 5.0, 7.0));

  MeanResponse const mr;
  auto leaf = mr.create_leaf(f.ctx, f.rng);

  ASSERT_NE(leaf, nullptr);
  EXPECT_TRUE(is_leaf(leaf));
  EXPECT_FLOAT_EQ(static_cast<float>(leaf->response()), 4.0F);
}

TEST(MeanResponseLeaf, MeanOfSingleGroup) {
  NodeContextFixture f(MAT(Feature, rows(3), 1, 2, 3, 4, 5, 6), VEC(Outcome, 2.0, 4.0, 6.0));

  MeanResponse const mr;
  auto leaf = mr.create_leaf(f.ctx, f.rng);

  EXPECT_FLOAT_EQ(static_cast<float>(leaf->response()), 4.0F);
}

TEST(MeanResponseLeaf, MeanOfSubset) {
  // Verifies MeanResponse averages only over rows the partition covers,
  // not the full y_vec — the partition `gp.subset({0})` is narrower than
  // the row range of x, and the fixture's `(x, GroupPartition, y_vec)`
  // ctor lets us pass that subset directly.
  FeatureMatrix x       = MAT(Feature, rows(4), 1, 2, 3, 4, 5, 6, 7, 8);
  GroupIdVector const y = VEC(GroupId, 0, 0, 1, 1);
  GroupPartition const gp(y);

  NodeContextFixture f(x, gp.subset({0}), VEC(Outcome, 10.0, 20.0, 30.0, 40.0));

  MeanResponse const mr;
  auto leaf = mr.create_leaf(f.ctx, f.rng);

  // Only group 0 observations (indices 0,1) → mean of 10.0 and 20.0 = 15.0
  EXPECT_FLOAT_EQ(static_cast<float>(leaf->response()), 15.0F);
}
