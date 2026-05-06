/**
 * @file VariableSelection.test.cpp
 * @brief Tests for the `VariableSelection` base-class NVI: `ctx.aborted`
 *        gating and the post-compute invariant on `ctx.var_selection`.
 *        Concrete subclasses (`All`, `Uniform`) are exercised elsewhere.
 */
#include <gtest/gtest.h>

#include "models/strategies/vars/VariableSelection.hpp"
#include "models/strategies/NodeContext.hpp"
#include "test/NodeContextFixture.hpp"
#include "stats/Stats.hpp"
#include "utils/Macros.hpp"
#include "utils/Types.hpp"

#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace ppforest2;
using namespace ppforest2::vars;
using namespace ppforest2::stats;
using namespace ppforest2::test;
using namespace ppforest2::types;

namespace {
  class MockVars : public VariableSelection {
  public:
    using ComputeFn = std::function<void(NodeContext&, RNG&)>;

    ComputeFn on_compute;
    mutable int compute_calls = 0;

    explicit MockVars(ComputeFn on_compute = {})
        : on_compute(std::move(on_compute)) {}

    nlohmann::json to_json() const override { return {{"name", "mock_vars"}}; }
    std::string display_name() const override { return "Mock Vars"; }
    std::set<types::Mode> supported_modes() const override {
      return {types::Mode::Classification, types::Mode::Regression};
    }

  protected:
    void compute(NodeContext& ctx, RNG& rng) const override {
      ++compute_calls;
      if (on_compute) {
        on_compute(ctx, rng);
      }
    }
  };
}

TEST(VariableSelectionNVI, SkipsComputeWhenAlreadyAborted) {
  NodeContextFixture f(MAT(Feature, rows(2), 1, 0, 0, 1), VEC(GroupId, 0, 1));
  f.ctx.aborted = true;

  MockVars strategy;
  strategy.select(f.ctx, f.rng);

  EXPECT_EQ(strategy.compute_calls, 0);
}

TEST(VariableSelectionNVI, ThrowsIfVarSelectionNotSet) {
  NodeContextFixture f(MAT(Feature, rows(2), 1, 0, 0, 1), VEC(GroupId, 0, 1));

  MockVars strategy([](NodeContext&, RNG&) {});
  EXPECT_THROW(strategy.select(f.ctx, f.rng), std::runtime_error);
}

TEST(VariableSelectionNVI, ComputeAbortSkipsInvariant) {
  NodeContextFixture f(MAT(Feature, rows(2), 1, 0, 0, 1), VEC(GroupId, 0, 1));

  MockVars strategy([](NodeContext& ctx, RNG&) { ctx.aborted = true; });

  EXPECT_NO_THROW(strategy.select(f.ctx, f.rng));
  EXPECT_FALSE(f.ctx.var_selection.has_value());
}

TEST(VariableSelectionNVI, HappyPathSetsContext) {
  NodeContextFixture f(MAT(Feature, rows(2), 1, 0, 0, 1), VEC(GroupId, 0, 1));

  MockVars strategy([](NodeContext& ctx, RNG&) {
    ctx.var_selection = VariableSelection::Result(std::vector<int>{0, 1}, 2);
  });
  strategy.select(f.ctx, f.rng);

  EXPECT_FALSE(f.ctx.aborted);
  ASSERT_TRUE(f.ctx.var_selection.has_value());
  EXPECT_EQ(f.ctx.var_selection->selected_cols.size(), 2U);
}
