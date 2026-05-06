#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "models/strategies/cutpoint/MeanOfMeans.hpp"
#include "models/strategies/cutpoint/Cutpoint.hpp"
#include "models/strategies/NodeContext.hpp"
#include "test/NodeContextFixture.hpp"
#include "utils/Types.hpp"
#include "utils/Macros.hpp"

using namespace ppforest2;
using namespace ppforest2::cutpoint;
using namespace ppforest2::pp;
using namespace ppforest2::stats;
using namespace ppforest2::test;
using namespace ppforest2::types;
using json = nlohmann::json;


TEST(CutpointMeanOfMeansStrategy, FromJsonValid) {
  json const j = {{"name", "mean_of_means"}};

  auto strategy = MeanOfMeans::from_json(j);

  ASSERT_NE(strategy, nullptr);
}

TEST(CutpointMeanOfMeansStrategy, FromJsonRoundTrip) {
  json const j = {{"name", "mean_of_means"}};

  auto strategy = MeanOfMeans::from_json(j);

  EXPECT_EQ(strategy->to_json(), j);
}

TEST(CutpointMeanOfMeansStrategy, FromJsonUnknownParam) {
  json const j = {{"name", "mean_of_means"}, {"extra", 1}};
  EXPECT_THROW(MeanOfMeans::from_json(j), std::runtime_error);
}

TEST(CutpointMeanOfMeansStrategy, RegistryLookup) {
  json const j = {{"name", "mean_of_means"}};

  auto strategy = Cutpoint::from_json(j);

  ASSERT_NE(strategy, nullptr);
  EXPECT_EQ(strategy->to_json(), j);
}

TEST(CutpointMeanOfMeansStrategy, RegistryUnknownStrategy) {
  json const j = {{"name", "unknown_cutpoint"}};
  EXPECT_THROW(Cutpoint::from_json(j), std::runtime_error);
}

TEST(CutpointMeanOfMeansStrategy, CutpointIsMidpointOfProjectedMeans) {
  // Group 0 rows are [1,2] and [3,4], mean = [2, 3]
  // Group 1 rows are [5,6] and [7,8], mean = [6, 7]
  // Projector [1, 0] → projected means 2 and 6 → cutpoint = (2 + 6) / 2 = 4.
  NodeContextFixture f(MAT(Feature, rows(4), 1, 2, 3, 4, 5, 6, 7, 8), VEC(GroupId, 0, 0, 1, 1));
  f.ctx.projector = VEC(Feature, 1, 0);

  MeanOfMeans().locate(f.ctx, f.rng);

  EXPECT_FLOAT_EQ(f.ctx.cutpoint.value(), 4.0F);
}

TEST(CutpointMeanOfMeansStrategy, CutpointWithNonTrivialProjector) {
  // Group 0 rows [1,0], [0,1] → mean = [0.5, 0.5]
  // Group 1 rows [4,0], [0,4] → mean = [2, 2]
  // Projector [1, 1] → projected means 1.0 and 4.0 → cutpoint = 2.5.
  NodeContextFixture f(MAT(Feature, rows(4), 1, 0, 0, 1, 4, 0, 0, 4), VEC(GroupId, 0, 0, 1, 1));
  f.ctx.projector = VEC(Feature, 1, 1);

  MeanOfMeans().locate(f.ctx, f.rng);

  EXPECT_FLOAT_EQ(f.ctx.cutpoint.value(), 2.5F);
}

TEST(CutpointMeanOfMeansStrategy, CutpointSymmetric) {
  // Re-labelling the groups (0 ↔ 1) should give the same cutpoint:
  // MeanOfMeans averages the two projected means in symmetric formula.
  FeatureMatrix const x = MAT(Feature, rows(4), 1, 2, 3, 4, 5, 6, 7, 8);
  Projector const proj  = VEC(Feature, 0, 1);

  NodeContextFixture f1(x, VEC(GroupId, 0, 0, 1, 1));
  NodeContextFixture f2(x, VEC(GroupId, 1, 1, 0, 0));

  f1.ctx.projector = proj;
  f2.ctx.projector = proj;

  MeanOfMeans().locate(f1.ctx, f1.rng);
  MeanOfMeans().locate(f2.ctx, f2.rng);

  EXPECT_FLOAT_EQ(f1.ctx.cutpoint.value(), f2.ctx.cutpoint.value());
}

TEST(CutpointMeanOfMeansStrategy, ComputeIsDeterministic) {
  FeatureMatrix const x = MAT(Feature, rows(4), 1, 2, 3, 4, 5, 6, 7, 8);
  Projector const proj  = VEC(Feature, 1, 0);

  NodeContextFixture f1(x, VEC(GroupId, 0, 0, 1, 1));
  NodeContextFixture f2(x, VEC(GroupId, 0, 0, 1, 1));

  f1.ctx.projector = proj;
  f2.ctx.projector = proj;

  MeanOfMeans().locate(f1.ctx, f1.rng);
  MeanOfMeans().locate(f2.ctx, f2.rng);

  EXPECT_FLOAT_EQ(f1.ctx.cutpoint.value(), f2.ctx.cutpoint.value());
}
