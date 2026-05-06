/**
 * @file Grouping.test.cpp
 * @brief Tests for the `Grouping` base-class NVI: `ctx.aborted` gating,
 *        the pre-compute invariants on `ctx.lower_group` /
 *        `ctx.upper_group`, the post-compute invariants on
 *        `ctx.lower_y_part` / `ctx.upper_y_part`, and the no-progress
 *        abort. Concrete subclasses (`ByLabel`, `ByCutpoint`) are
 *        exercised elsewhere.
 */
#include <gtest/gtest.h>

#include "models/strategies/grouping/Grouping.hpp"
#include "models/strategies/NodeContext.hpp"
#include "test/NodeContextFixture.hpp"
#include "stats/Stats.hpp"
#include "utils/Macros.hpp"
#include "utils/Types.hpp"

#include <functional>
#include <stdexcept>
#include <utility>

using namespace ppforest2;
using namespace ppforest2::grouping;
using namespace ppforest2::stats;
using namespace ppforest2::test;
using namespace ppforest2::types;

namespace {
  class MockGrouping : public Grouping {
  public:
    using ComputeFn = std::function<void(NodeContext&, RNG&)>;

    ComputeFn on_compute;
    mutable int compute_calls = 0;

    explicit MockGrouping(ComputeFn on_compute = {})
        : on_compute(std::move(on_compute)) {}

    nlohmann::json to_json() const override { return {{"name", "mock_grouping"}}; }
    std::string display_name() const override { return "Mock Grouping"; }
    std::set<types::Mode> supported_modes() const override {
      return {types::Mode::Classification, types::Mode::Regression};
    }

  protected:
    GroupPartition compute_init(OutcomeVector const& y) const override {
      GroupIdVector const ids = y.cast<GroupId>();
      return GroupPartition(ids);
    }

    void compute(NodeContext& ctx, RNG& rng) const override {
      ++compute_calls;
      if (on_compute) {
        on_compute(ctx, rng);
      }
    }
  };
}

TEST(GroupingNVI, SkipsComputeWhenAlreadyAborted) {
  NodeContextFixture f(MAT(Feature, rows(4), 1, 0, 0, 1, 1, 1, 0, 0), VEC(GroupId, 0, 0, 1, 1));
  f.ctx.lower_group = 0;
  f.ctx.upper_group = 1;
  f.ctx.aborted     = true;

  MockGrouping strategy;
  strategy.split(f.ctx, f.rng);

  EXPECT_EQ(strategy.compute_calls, 0);
}

TEST(GroupingNVI, ThrowsIfLowerGroupNotSet) {
  NodeContextFixture f(MAT(Feature, rows(4), 1, 0, 0, 1, 1, 1, 0, 0), VEC(GroupId, 0, 0, 1, 1));
  f.ctx.upper_group = 1;
  // lower_group intentionally left unset

  MockGrouping strategy;
  EXPECT_THROW(strategy.split(f.ctx, f.rng), std::runtime_error);
}

TEST(GroupingNVI, ThrowsIfUpperGroupNotSet) {
  NodeContextFixture f(MAT(Feature, rows(4), 1, 0, 0, 1, 1, 1, 0, 0), VEC(GroupId, 0, 0, 1, 1));
  f.ctx.lower_group = 0;
  // upper_group intentionally left unset

  MockGrouping strategy;
  EXPECT_THROW(strategy.split(f.ctx, f.rng), std::runtime_error);
}

TEST(GroupingNVI, ThrowsIfLowerYPartNotSet) {
  NodeContextFixture f(MAT(Feature, rows(4), 1, 0, 0, 1, 1, 1, 0, 0), VEC(GroupId, 0, 0, 1, 1));
  f.ctx.lower_group = 0;
  f.ctx.upper_group = 1;

  // sets only the upper child
  MockGrouping strategy([](NodeContext& ctx, RNG&) { ctx.upper_y_part.emplace(ctx.y); });
  EXPECT_THROW(strategy.split(f.ctx, f.rng), std::runtime_error);
}

TEST(GroupingNVI, ThrowsIfUpperYPartNotSet) {
  NodeContextFixture f(MAT(Feature, rows(4), 1, 0, 0, 1, 1, 1, 0, 0), VEC(GroupId, 0, 0, 1, 1));
  f.ctx.lower_group = 0;
  f.ctx.upper_group = 1;

  // sets only the lower child
  MockGrouping strategy([](NodeContext& ctx, RNG&) { ctx.lower_y_part.emplace(ctx.y); });
  EXPECT_THROW(strategy.split(f.ctx, f.rng), std::runtime_error);
}

TEST(GroupingNVI, AbortsOnNoProgress) {
  // Lower child receives every row (its size == parent size) — the NVI
  // detects that recursion would loop and aborts.
  NodeContextFixture f(MAT(Feature, rows(4), 1, 0, 0, 1, 1, 1, 0, 0), VEC(GroupId, 0, 0, 1, 1));
  f.ctx.lower_group = 0;
  f.ctx.upper_group = 1;

  MockGrouping strategy([](NodeContext& ctx, RNG&) {
    ctx.lower_y_part.emplace(ctx.y);                // full parent partition
    ctx.upper_y_part.emplace(GroupPartition(0, 0)); // 1-row, won't matter
  });
  strategy.split(f.ctx, f.rng);

  EXPECT_TRUE(f.ctx.aborted);
}

TEST(GroupingNVI, HappyPathDoesNotAbort) {
  // Each child receives a strict subset of the parent's rows.
  NodeContextFixture f(MAT(Feature, rows(4), 1, 0, 0, 1, 1, 1, 0, 0), VEC(GroupId, 0, 0, 1, 1));
  f.ctx.lower_group = 0;
  f.ctx.upper_group = 1;

  MockGrouping strategy([](NodeContext& ctx, RNG&) {
    GroupIdVector const lower_ids = VEC(GroupId, 0, 0);
    GroupIdVector const upper_ids = VEC(GroupId, 1, 1);
    ctx.lower_y_part.emplace(GroupPartition(lower_ids));
    ctx.upper_y_part.emplace(GroupPartition(upper_ids));
  });
  strategy.split(f.ctx, f.rng);

  EXPECT_FALSE(f.ctx.aborted);
  EXPECT_TRUE(f.ctx.lower_y_part.has_value());
  EXPECT_TRUE(f.ctx.upper_y_part.has_value());
}
