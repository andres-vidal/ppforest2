/**
 * @file ProjectionPursuit.test.cpp
 * @brief Tests for the `ProjectionPursuit` base-class NVI: `ctx.aborted`
 *        gating, post-compute invariants on `ctx.projector` and
 *        `ctx.pp_index_value`, and the NaN-projector abort. Concrete
 *        subclasses (`PDA`, ...) are exercised elsewhere.
 */
#include <gtest/gtest.h>

#include "models/strategies/pp/ProjectionPursuit.hpp"
#include "models/strategies/NodeContext.hpp"
#include "test/NodeContextFixture.hpp"
#include "stats/Stats.hpp"
#include "utils/Macros.hpp"
#include "utils/Types.hpp"

#include <functional>
#include <limits>
#include <stdexcept>
#include <utility>

using namespace ppforest2;
using namespace ppforest2::pp;
using namespace ppforest2::stats;
using namespace ppforest2::test;
using namespace ppforest2::types;

namespace {
  // Configurable mock strategy: each test passes the `compute` body it
  // needs straight into the constructor.
  class MockPP : public ProjectionPursuit {
  public:
    using ComputeFn = std::function<void(NodeContext&, RNG&)>;

    ComputeFn on_compute;
    mutable int compute_calls = 0;

    explicit MockPP(ComputeFn on_compute = {})
        : on_compute(std::move(on_compute)) {}

    nlohmann::json to_json() const override { return {{"name", "mock_pp"}}; }
    std::string display_name() const override { return "Mock PP"; }
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

TEST(ProjectionPursuitNVI, SkipsComputeWhenAlreadyAborted) {
  NodeContextFixture f(MAT(Feature, rows(2), 1, 0, 0, 1), VEC(GroupId, 0, 1));
  f.ctx.aborted = true;

  MockPP strategy;
  strategy.optimize(f.ctx, f.rng);

  EXPECT_EQ(strategy.compute_calls, 0);
}

TEST(ProjectionPursuitNVI, ThrowsIfProjectorNotSet) {
  NodeContextFixture f(MAT(Feature, rows(2), 1, 0, 0, 1), VEC(GroupId, 0, 1));

  MockPP strategy([](NodeContext& ctx, RNG&) { ctx.pp_index_value = 1; });
  EXPECT_THROW(strategy.optimize(f.ctx, f.rng), std::runtime_error);
}

TEST(ProjectionPursuitNVI, ThrowsIfIndexValueNotSet) {
  NodeContextFixture f(MAT(Feature, rows(2), 1, 0, 0, 1), VEC(GroupId, 0, 1));

  MockPP strategy([](NodeContext& ctx, RNG&) { ctx.projector = Projector(VEC(Feature, 1, 0)); });

  EXPECT_THROW(strategy.optimize(f.ctx, f.rng), std::runtime_error);
}

TEST(ProjectionPursuitNVI, AbortsIfProjectorHasNaN) {
  NodeContextFixture f(MAT(Feature, rows(2), 1, 0, 0, 1), VEC(GroupId, 0, 1));

  MockPP strategy([](NodeContext& ctx, RNG&) {
    Projector p(2);
    p << std::numeric_limits<Feature>::quiet_NaN(), 0;
    ctx.projector      = p;
    ctx.pp_index_value = 0.5F;
  });
  strategy.optimize(f.ctx, f.rng);

  EXPECT_TRUE(f.ctx.aborted);
}

TEST(ProjectionPursuitNVI, ComputeAbortSkipsInvariants) {
  // If the subclass sets ctx.aborted itself, the NVI must short-circuit
  // before checking the post-compute invariants — otherwise a clean abort
  // path would throw on missing projector/index_value.
  NodeContextFixture f(MAT(Feature, rows(2), 1, 0, 0, 1), VEC(GroupId, 0, 1));

  MockPP strategy([](NodeContext& ctx, RNG&) { ctx.aborted = true; });
  EXPECT_NO_THROW(strategy.optimize(f.ctx, f.rng));
  EXPECT_FALSE(f.ctx.projector.has_value());
}

TEST(ProjectionPursuitNVI, HappyPathSetsContextAndDoesNotAbort) {
  NodeContextFixture f(MAT(Feature, rows(2), 1, 0, 0, 1), VEC(GroupId, 0, 1));

  MockPP strategy([](NodeContext& ctx, RNG&) {
    ctx.projector      = Projector(VEC(Feature, 1, 0));
    ctx.pp_index_value = 0.7F;
  });
  strategy.optimize(f.ctx, f.rng);

  EXPECT_FALSE(f.ctx.aborted);
  ASSERT_TRUE(f.ctx.projector.has_value());
  EXPECT_FLOAT_EQ(f.ctx.pp_index_value.value(), 0.7F);
}
