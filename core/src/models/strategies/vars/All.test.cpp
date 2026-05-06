#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "models/strategies/vars/All.hpp"
#include "models/strategies/vars/VariableSelection.hpp"
#include "models/strategies/NodeContext.hpp"
#include "test/NodeContextFixture.hpp"
#include "stats/GroupPartition.hpp"
#include "stats/Stats.hpp"
#include "utils/Types.hpp"
#include "utils/Macros.hpp"

using namespace ppforest2;
using namespace ppforest2::vars;
using namespace ppforest2::stats;
using namespace ppforest2::test;
using namespace ppforest2::types;
using json = nlohmann::json;

TEST(VarsAllStrategy, FromJsonValid) {
  json const j  = {{"name", "all"}};
  auto strategy = All::from_json(j);
  ASSERT_NE(strategy, nullptr);
}

TEST(VarsAllStrategy, FromJsonRoundTrip) {
  json const j  = {{"name", "all"}};
  auto strategy = All::from_json(j);

  auto out = strategy->to_json();

  EXPECT_EQ(out, j);
}

TEST(VarsAllStrategy, FromJsonUnknownParam) {
  json const j = {{"name", "all"}, {"unexpected", 0}};
  EXPECT_THROW(All::from_json(j), std::runtime_error);
}

TEST(VarsAllStrategy, RegistryLookup) {
  json const j  = {{"name", "all"}};
  auto strategy = VariableSelection::from_json(j);
  ASSERT_NE(strategy, nullptr);

  auto out = strategy->to_json();
  EXPECT_EQ(out, j);
}

TEST(VarsAllStrategy, SelectsAllColumns) {
  NodeContextFixture f(MAT(Feature, rows(3), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), VEC(GroupId, 0, 1, 1));

  All const vs;
  vs.select(f.ctx, f.rng);

  ASSERT_TRUE(f.ctx.var_selection.has_value());
  EXPECT_EQ(f.ctx.var_selection->selected_cols.size(), 4U);
  EXPECT_EQ(f.ctx.var_selection->original_size, 4U);
  EXPECT_EQ(f.ctx.var_selection->selected_cols, (std::vector<int>{0, 1, 2, 3}));
}
