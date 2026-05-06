/**
 * @file Binarization.test.cpp
 * @brief Tests for the `Binarization` base-class NVI: `ctx.aborted`
 *        gating, the post-compute invariant on `ctx.y_bin`, and the
 *        abort-on-collapsed-binary guard. Concrete subclasses
 *        (`LargestGap`, `Disabled`, ...) are exercised elsewhere.
 */
#include <gtest/gtest.h>

#include "models/strategies/binarize/Binarization.hpp"
#include "models/strategies/NodeContext.hpp"
#include "test/NodeContextFixture.hpp"
#include "stats/Stats.hpp"
#include "utils/Macros.hpp"
#include "utils/Types.hpp"

#include <functional>
#include <map>
#include <stdexcept>
#include <utility>

using namespace ppforest2;
using namespace ppforest2::binarize;
using namespace ppforest2::stats;
using namespace ppforest2::test;
using namespace ppforest2::types;

namespace {
  class MockBinarize : public Binarization {
  public:
    using ComputeFn = std::function<void(NodeContext&, RNG&)>;

    ComputeFn on_compute;
    mutable int compute_calls = 0;

    explicit MockBinarize(ComputeFn on_compute = {})
        : on_compute(std::move(on_compute)) {}

    nlohmann::json to_json() const override { return {{"name", "mock_binarize"}}; }
    std::string display_name() const override { return "Mock Binarize"; }
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

TEST(BinarizationNVI, SkipsComputeWhenAlreadyAborted) {
  NodeContextFixture f(MAT(Feature, rows(3), 1, 0, 0, 1, 1, 1), VEC(GroupId, 0, 1, 2));
  f.ctx.aborted = true;

  MockBinarize strategy;
  strategy.regroup(f.ctx, f.rng);

  EXPECT_EQ(strategy.compute_calls, 0);
}

TEST(BinarizationNVI, ThrowsIfYBinNotSet) {
  NodeContextFixture f(MAT(Feature, rows(3), 1, 0, 0, 1, 1, 1), VEC(GroupId, 0, 1, 2));

  MockBinarize strategy([](NodeContext&, RNG&) {});

  EXPECT_THROW(strategy.regroup(f.ctx, f.rng), std::runtime_error);
}

TEST(BinarizationNVI, AbortsIfBinaryCollapsedToOneGroup) {
  // All three original groups remap to a single binary group → aborted.
  NodeContextFixture f(MAT(Feature, rows(3), 1, 0, 0, 1, 1, 1), VEC(GroupId, 0, 1, 2));

  MockBinarize strategy([](NodeContext& ctx, RNG&) {
    std::map<int, int> mapping = {{0, 0}, {1, 0}, {2, 0}};
    ctx.y_bin.emplace(ctx.y.remap(mapping));
  });

  strategy.regroup(f.ctx, f.rng);

  EXPECT_TRUE(f.ctx.aborted);
  ASSERT_TRUE(f.ctx.y_bin.has_value());
  EXPECT_EQ(f.ctx.y_bin->groups.size(), 1U);
}

TEST(BinarizationNVI, ComputeAbortSkipsInvariant) {
  NodeContextFixture f(MAT(Feature, rows(3), 1, 0, 0, 1, 1, 1), VEC(GroupId, 0, 1, 2));

  MockBinarize strategy([](NodeContext& ctx, RNG&) { ctx.aborted = true; });

  EXPECT_NO_THROW(strategy.regroup(f.ctx, f.rng));
  EXPECT_FALSE(f.ctx.y_bin.has_value());
}

TEST(BinarizationNVI, HappyPathSetsBinaryPartitionAndDoesNotAbort) {
  NodeContextFixture f(MAT(Feature, rows(3), 1, 0, 0, 1, 1, 1), VEC(GroupId, 0, 1, 2));

  MockBinarize strategy([](NodeContext& ctx, RNG&) {
    std::map<int, int> mapping = {{0, 0}, {1, 0}, {2, 1}};
    ctx.y_bin.emplace(ctx.y.remap(mapping));
  });
  strategy.regroup(f.ctx, f.rng);

  EXPECT_FALSE(f.ctx.aborted);
  ASSERT_TRUE(f.ctx.y_bin.has_value());
  EXPECT_EQ(f.ctx.y_bin->groups.size(), 2U);
}
