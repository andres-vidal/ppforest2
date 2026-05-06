#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "models/strategies/vars/Uniform.hpp"
#include "models/strategies/vars/VariableSelection.hpp"
#include "models/strategies/NodeContext.hpp"
#include "test/NodeContextFixture.hpp"
#include "stats/Stats.hpp"
#include "utils/Types.hpp"
#include "utils/Macros.hpp"

using namespace ppforest2;
using namespace ppforest2::vars;
using namespace ppforest2::stats;
using namespace ppforest2::test;
using namespace ppforest2::types;
using json = nlohmann::json;


TEST(VarsUniformStrategy, FromJsonValid) {
  json const j = {{"name", "uniform"}, {"count", 3}};

  auto strategy = Uniform::from_json(j);

  ASSERT_NE(strategy, nullptr);
}

TEST(VarsUniformStrategy, FromJsonRoundTrip) {
  json const j = {{"name", "uniform"}, {"count", 3}};

  auto strategy = Uniform::from_json(j);

  EXPECT_EQ(strategy->to_json(), j);
}

TEST(VarsUniformStrategy, FromJsonWithProportion) {
  json const j = {{"name", "uniform"}, {"proportion", 0.5}};

  auto strategy = Uniform::from_json(j);

  ASSERT_NE(strategy, nullptr);
}

TEST(VarsUniformStrategy, FromJsonRejectsInvalidProportion) {
  EXPECT_THROW(Uniform::from_json({{"name", "uniform"}, {"proportion", 1.5}}), std::exception);
  EXPECT_THROW(Uniform::from_json({{"name", "uniform"}, {"proportion", 0.0}}), std::exception);
  EXPECT_THROW(Uniform::from_json({{"name", "uniform"}, {"proportion", -0.5}}), std::exception);
}

TEST(VarsUniformStrategy, FromJsonMissingNVars) {
  json const j = {{"name", "uniform"}};
  EXPECT_THROW(Uniform::from_json(j), std::exception);
}

TEST(VarsUniformStrategy, FromJsonUnknownParam) {
  json const j = {{"name", "uniform"}, {"count", 3}, {"extra", true}};
  EXPECT_THROW(Uniform::from_json(j), std::runtime_error);
}

TEST(VarsUniformStrategy, RegistryLookup) {
  json const j = {{"name", "uniform"}, {"count", 2}};

  auto strategy = VariableSelection::from_json(j);

  ASSERT_NE(strategy, nullptr);
  EXPECT_EQ(strategy->to_json(), j);
}

TEST(VarsUniformStrategy, SelectsCorrectNumberOfVars) {
  FeatureMatrix const x = MAT(Feature, rows(4), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20);
  GroupIdVector const y = GroupIdVector::Zero(x.rows());
  NodeContextFixture f(x, y);

  Uniform(2).select(f.ctx, f.rng);

  auto const& result = f.ctx.var_selection.value();
  ASSERT_EQ(result.selected_cols.size(), 2U);
  EXPECT_EQ(result.original_size, 5U);
}

TEST(VarsUniformStrategy, AllVarsReturnsAllIndices) {
  FeatureMatrix const x = MAT(Feature, rows(4), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
  GroupIdVector const y = GroupIdVector::Zero(x.rows());
  NodeContextFixture f(x, y);

  Uniform(3).select(f.ctx, f.rng);

  auto const& result = f.ctx.var_selection.value();
  ASSERT_EQ(result.selected_cols.size(), 3U);
  EXPECT_EQ(result.selected_cols, (std::vector<int>{0, 1, 2}));
}

TEST(VarsUniformStrategy, RejectsZeroVars) {
  EXPECT_THROW(Uniform(0), std::exception);
}

TEST(VarsUniformStrategy, DeterministicWithSameSeed) {
  FeatureMatrix const x = MAT(Feature, rows(4), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20);
  GroupIdVector const y = GroupIdVector::Zero(x.rows());
  NodeContextFixture f1(x, y);
  NodeContextFixture f2(x, y);

  f1.rng.seed(0);
  f2.rng.seed(0);

  Uniform const vs(2);

  vs.select(f1.ctx, f1.rng);
  vs.select(f2.ctx, f2.rng);

  EXPECT_EQ(f1.ctx.var_selection->selected_cols, f2.ctx.var_selection->selected_cols);
}

TEST(VarsUniformStrategy, DifferentWithDifferentSeed) {
  FeatureMatrix const x = MAT(Feature, rows(4), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20);
  GroupIdVector const y = GroupIdVector::Zero(x.rows());
  NodeContextFixture f1(x, y);
  NodeContextFixture f2(x, y);

  f1.rng.seed(1);
  f2.rng.seed(2);

  Uniform const vs(2);

  vs.select(f1.ctx, f1.rng);
  vs.select(f2.ctx, f2.rng);

  EXPECT_NE(f1.ctx.var_selection->selected_cols, f2.ctx.var_selection->selected_cols);
}
