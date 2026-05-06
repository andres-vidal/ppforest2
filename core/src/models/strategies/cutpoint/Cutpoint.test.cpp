/**
 * @file Cutpoint.test.cpp
 * @brief Tests for the `Cutpoint` base-class NVI: `ctx.aborted` gating,
 *        the post-compute invariant on `ctx.cutpoint`, the 2-group
 *        precondition on the active partition, and the
 *        orient-by-projected-mean step that writes `ctx.lower_group` /
 *        `ctx.upper_group`. Concrete subclasses (`MeanOfMeans`, ...) are
 *        exercised elsewhere.
 */
#include <gtest/gtest.h>

#include "models/strategies/cutpoint/Cutpoint.hpp"
#include "models/strategies/NodeContext.hpp"
#include "test/NodeContextFixture.hpp"
#include "stats/Stats.hpp"
#include "utils/Macros.hpp"
#include "utils/Types.hpp"

#include <functional>
#include <stdexcept>

using namespace ppforest2;
using namespace ppforest2::cutpoint;
using namespace ppforest2::pp;
using namespace ppforest2::stats;
using namespace ppforest2::test;
using namespace ppforest2::types;

namespace {
  class MockCutpoint : public Cutpoint {
  public:
    using ComputeFn = std::function<void(NodeContext&, RNG&)>;

    ComputeFn on_compute;
    mutable int compute_calls = 0;

    explicit MockCutpoint(ComputeFn on_compute = {})
        : on_compute(std::move(on_compute)) {}

    nlohmann::json to_json() const override { return {{"name", "mock_cp"}}; }
    std::string display_name() const override { return "Mock Cutpoint"; }
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

TEST(CutpointNVI, SkipsComputeWhenAlreadyAborted) {
  NodeContextFixture f(MAT(Feature, rows(2), 1, 0, 0, 1), VEC(GroupId, 0, 1));
  f.ctx.projector = Projector(VEC(Feature, 1, 0));
  f.ctx.aborted   = true;

  MockCutpoint strategy;
  strategy.locate(f.ctx, f.rng);

  EXPECT_EQ(strategy.compute_calls, 0);
}

TEST(CutpointNVI, ThrowsIfCutpointNotSet) {
  NodeContextFixture f(MAT(Feature, rows(2), 1, 0, 0, 1), VEC(GroupId, 0, 1));
  f.ctx.projector = Projector(VEC(Feature, 1, 0));

  MockCutpoint strategy([](NodeContext&, RNG&) {});

  EXPECT_THROW(strategy.locate(f.ctx, f.rng), std::runtime_error);
}

TEST(CutpointNVI, ThrowsIfPartitionDoesNotHaveTwoGroups) {
  NodeContextFixture f(MAT(Feature, rows(2), 1, 0, 0, 1), VEC(GroupId, 0, 0));
  f.ctx.projector = Projector(VEC(Feature, 1, 0));

  MockCutpoint strategy([](NodeContext& c, RNG&) { c.cutpoint = 0.5F; });

  EXPECT_THROW(strategy.locate(f.ctx, f.rng), std::runtime_error);
}

TEST(CutpointNVI, ThrowsIfThereIsNoProjector) {
  NodeContextFixture f(MAT(Feature, rows(2), 1, 0, 0, 1), VEC(GroupId, 0, 0));

  MockCutpoint strategy([](NodeContext& ctx, RNG&) { ctx.cutpoint = 0.5F; });

  EXPECT_THROW(strategy.locate(f.ctx, f.rng), std::runtime_error);
}

TEST(CutpointNVI, ThrowsIfThereIsStrategyClearsProjector) {
  NodeContextFixture f(MAT(Feature, rows(2), 1, 0, 0, 1), VEC(GroupId, 0, 0));
  f.ctx.projector = Projector(VEC(Feature, 1, 0));

  MockCutpoint strategy([](NodeContext& ctx, RNG&) {
    ctx.cutpoint  = 0.5F;
    ctx.projector = std::nullopt;
  });

  EXPECT_THROW(strategy.locate(f.ctx, f.rng), std::runtime_error);
}

TEST(CutpointNVI, ComputeAbortSkipsInvariantsAndOrient) {
  NodeContextFixture f(MAT(Feature, rows(2), 1, 0, 0, 1), VEC(GroupId, 0, 1));
  f.ctx.projector = Projector(VEC(Feature, 1, 0));

  MockCutpoint strategy([](NodeContext& ctx, RNG&) { ctx.aborted = true; });

  EXPECT_NO_THROW(strategy.locate(f.ctx, f.rng));
  EXPECT_FALSE(f.ctx.cutpoint.has_value());
  EXPECT_FALSE(f.ctx.lower_group.has_value());
  EXPECT_FALSE(f.ctx.upper_group.has_value());
}

TEST(CutpointNVI, OrientsLowerAndUpperByProjectedMean) {
  // Group 0's row is at (0, 0), group 1's at (100, 0). Under the
  // projector [1, 0] the projected means are 0 and 100 respectively, so
  // orient must put group 0 in `lower_group` and group 1 in `upper_group`.
  NodeContextFixture f(MAT(Feature, rows(2), 0, 0, 100, 0), VEC(GroupId, 0, 1));
  f.ctx.projector = Projector(VEC(Feature, 1, 0));

  MockCutpoint strategy([](NodeContext& c, RNG&) { c.cutpoint = 50.0F; });

  strategy.locate(f.ctx, f.rng);

  ASSERT_TRUE(f.ctx.lower_group.has_value());
  ASSERT_TRUE(f.ctx.upper_group.has_value());
  EXPECT_EQ(f.ctx.lower_group.value(), 0);
  EXPECT_EQ(f.ctx.upper_group.value(), 1);
  EXPECT_FLOAT_EQ(f.ctx.cutpoint.value(), 50.0F);
}

TEST(CutpointNVI, OrientsCorrectlyWhenSwappedByProjector) {
  // Same row geometry as the previous test, but the projector points the
  // opposite way: group 1's projected mean is now -100 and group 0's is
  // 0, so the orient step must swap them.
  NodeContextFixture f(MAT(Feature, rows(2), 0, 0, 100, 0), VEC(GroupId, 0, 1));
  f.ctx.projector = Projector(VEC(Feature, -1, 0));

  MockCutpoint strategy([](NodeContext& c, RNG&) { c.cutpoint = -50.0F; });

  strategy.locate(f.ctx, f.rng);

  ASSERT_TRUE(f.ctx.lower_group.has_value());
  ASSERT_TRUE(f.ctx.upper_group.has_value());
  EXPECT_EQ(f.ctx.lower_group.value(), 1);
  EXPECT_EQ(f.ctx.upper_group.value(), 0);
}
