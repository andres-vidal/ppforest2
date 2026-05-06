#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "models/strategies/pp/ProjectionPursuit.hpp"
#include "models/strategies/pp/PDA.hpp"
#include "models/strategies/vars/VariableSelection.hpp"
#include "models/strategies/NodeContext.hpp"
#include "test/NodeContextFixture.hpp"
#include "utils/RangeVector.hpp"
#include "utils/Types.hpp"
#include "utils/Macros.hpp"

using namespace ppforest2;
using namespace ppforest2::pp;
using namespace ppforest2::stats;
using namespace ppforest2::test;
using namespace ppforest2::types;
using json = nlohmann::json;

namespace {
  class PDAFixture : public NodeContextFixture {
  public:
    PDAFixture(FeatureMatrix x_in, GroupIdVector ids_in)
        : NodeContextFixture(std::move(x_in), std::move(ids_in)) {
      ctx.var_selection =
          vars::VariableSelection::Result(utils::range_vector(x.cols()), static_cast<std::size_t>(x.cols()));
    }

    PDAFixture(FeatureMatrix x_in, GroupIdVector ids_in, std::vector<int> const& selected_cols)
        : NodeContextFixture(std::move(x_in), std::move(ids_in)) {
      ctx.var_selection = vars::VariableSelection::Result(selected_cols, static_cast<std::size_t>(x.cols()));
    }
  };

  // 3-column data with parallel signals in cols 0 and 2 and uniform noise in
  // col 1. The two subset tests below flip which signal column lives inside
  // `var_selection` — PDA must give the selected signal column non-zero
  // weight, and `Result::expand` must zero the columns left out of the
  // selection.
  namespace {
    inline FeatureMatrix two_signal_x() {
      return MAT(
          Feature, rows(6), 0.0, 1.0, 0.0, 0.1, 2.0, 0.1, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.1, 2.0, 1.1, 1.0, 1.0, 1.0
      );
    }

    inline GroupIdVector two_signal_y() {
      return VEC(GroupId, 0, 0, 0, 1, 1, 1);
    }
  }
}

TEST(PPPDAStrategy, FromJsonValid) {
  json const j  = {{"name", "pda"}, {"lambda", 0.3F}};
  auto strategy = PDA::from_json(j);
  ASSERT_NE(strategy, nullptr);
}

TEST(PPPDAStrategy, FromJsonRoundTrip) {
  json const j = {{"name", "pda"}, {"lambda", 0.3F}};

  auto strategy = PDA::from_json(j);

  EXPECT_EQ(strategy->to_json(), j);
}

TEST(PPPDAStrategy, FromJsonMissingLambda) {
  json const j = {{"name", "pda"}};
  EXPECT_THROW(PDA::from_json(j), std::exception);
}

TEST(PPPDAStrategy, FromJsonUnknownParam) {
  json const j = {{"name", "pda"}, {"lambda", 0.3F}, {"unknown", 1}};
  EXPECT_THROW(PDA::from_json(j), std::runtime_error);
}

TEST(PPPDAStrategy, RegistryLookup) {
  json const j = {{"name", "pda"}, {"lambda", 0.5F}};

  auto strategy = ProjectionPursuit::from_json(j);

  ASSERT_NE(strategy, nullptr);
  EXPECT_EQ(strategy->to_json(), j);
}

TEST(PPPDAStrategy, RegistryUnknownStrategy) {
  json const j = {{"name", "unknown_strategy"}};
  EXPECT_THROW(ProjectionPursuit::from_json(j), std::runtime_error);
}


TEST(Projector, LDAOptimumProjectorTwoGroups1) {
  FeatureMatrix x =
      MAT(Feature,
          rows(10),
          1,
          0,
          1,
          1,
          1,
          1,
          0,
          0,
          1,
          0,
          0,
          1,
          1,
          1,
          1,
          1,
          4,
          0,
          0,
          1,
          4,
          0,
          0,
          2,
          4,
          0,
          0,
          3,
          4,
          1,
          0,
          1,
          4,
          0,
          1,
          1,
          4,
          0,
          1,
          2);

  OutcomeVector y = VEC(Outcome, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);

  GroupIdVector const y_ids = y.cast<GroupId>();
  PDAFixture f(x, y_ids);
  PDA(0).optimize(f.ctx, f.rng);

  auto const& actual = f.ctx.projector.value();
  auto const index   = f.ctx.pp_index_value.value();

  ASSERT_COLLINEAR(actual, VEC(Feature, -1, 0, 0, 0));
  ASSERT_FLOAT_EQ(index, 1.0F);
}

TEST(Projector, LDAOptimumProjectorTwoGroups2) {
  FeatureMatrix x =
      MAT(Feature,
          rows(10),
          0,
          1,
          1,
          1,
          1,
          1,
          0,
          0,
          0,
          1,
          0,
          1,
          1,
          1,
          1,
          1,
          0,
          4,
          0,
          1,
          0,
          4,
          0,
          2,
          0,
          4,
          0,
          3,
          1,
          4,
          0,
          1,
          0,
          4,
          1,
          1,
          0,
          4,
          1,
          2);


  GroupIdVector const y = VEC(GroupId, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);

  PDAFixture f(x, y);
  PDA(0).optimize(f.ctx, f.rng);

  auto const& actual = f.ctx.projector.value();
  auto const index   = f.ctx.pp_index_value.value();

  FeatureVector const expected = VEC(Feature, 0, 1, 0, 0);


  ASSERT_COLLINEAR(expected, actual);
  ASSERT_FLOAT_EQ(1.0F, index) << "Optimal LDA projector for two groups has index 1";
}

TEST(Projector, LDAOptimumProjectorTwoGroups3) {
  FeatureMatrix x =
      MAT(Feature,
          rows(10),
          0,
          1,
          1,
          1,
          1,
          0,
          1,
          0,
          0,
          0,
          1,
          1,
          1,
          1,
          1,
          1,
          0,
          0,
          4,
          1,
          0,
          0,
          4,
          2,
          0,
          0,
          4,
          3,
          1,
          0,
          4,
          1,
          0,
          1,
          4,
          1,
          0,
          1,
          4,
          2);


  GroupIdVector const y = VEC(GroupId, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);

  PDAFixture f(x, y);
  PDA(0).optimize(f.ctx, f.rng);

  auto const& actual = f.ctx.projector.value();
  auto const index   = f.ctx.pp_index_value.value();

  FeatureVector const expected = VEC(Feature, 0, 0, -1, 0);

  ASSERT_COLLINEAR(expected, actual);
  ASSERT_FLOAT_EQ(1.0F, index) << "Optimal LDA projector for two groups has index 1";
}

TEST(Projector, LDAOptimumProjectorTwoGroups4) {
  FeatureMatrix x =
      MAT(Feature,
          rows(10),
          0,
          1,
          1,
          1,
          1,
          0,
          0,
          1,
          0,
          0,
          1,
          1,
          1,
          1,
          1,
          1,
          0,
          0,
          1,
          4,
          0,
          0,
          2,
          4,
          0,
          0,
          3,
          4,
          1,
          0,
          1,
          4,
          0,
          1,
          1,
          4,
          0,
          1,
          2,
          4);

  GroupIdVector const y = VEC(GroupId, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);

  PDAFixture f(x, y);
  PDA(0).optimize(f.ctx, f.rng);

  auto const& actual = f.ctx.projector.value();
  auto const index   = f.ctx.pp_index_value.value();

  FeatureVector const expected =
      VEC(Feature, 2.0965219514666735e-15, 4.4408920985006262e-16, -2.4980018054066002e-16, 1);

  ASSERT_COLLINEAR(expected, actual);
  ASSERT_FLOAT_EQ(1.0F, index) << "Optimal LDA projector for two groups has index 1";
}

TEST(Projector, LDAOptimumProjectorThreeGroups1) {
  FeatureMatrix x =
      MAT(Feature,
          rows(30),
          1,
          0,
          0,
          1,
          1,
          1,
          0,
          1,
          0,
          0,
          1,
          0,
          0,
          0,
          1,
          1,
          0,
          1,
          1,
          1,
          1,
          0,
          0,
          1,
          1,
          1,
          0,
          1,
          1,
          0,
          1,
          0,
          0,
          1,
          1,
          1,
          0,
          1,
          1,
          2,
          1,
          0,
          0,
          2,
          0,
          1,
          0,
          2,
          1,
          0,
          2,
          8,
          0,
          0,
          1,
          2,
          8,
          0,
          0,
          2,
          2,
          8,
          1,
          0,
          2,
          2,
          8,
          1,
          0,
          1,
          2,
          8,
          0,
          1,
          1,
          2,
          8,
          0,
          1,
          2,
          2,
          8,
          2,
          1,
          1,
          2,
          8,
          1,
          1,
          1,
          2,
          8,
          1,
          1,
          2,
          2,
          8,
          2,
          1,
          2,
          2,
          8,
          1,
          2,
          1,
          2,
          8,
          2,
          1,
          1,
          9,
          8,
          0,
          0,
          1,
          9,
          8,
          0,
          0,
          2,
          9,
          8,
          1,
          0,
          2,
          9,
          8,
          1,
          0,
          1,
          9,
          8,
          0,
          1,
          1,
          9,
          8,
          0,
          1,
          2,
          9,
          8,
          2,
          1,
          1,
          9,
          8,
          1,
          1,
          1);

  GroupIdVector const y =
      VEC(GroupId, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2);

  PDAFixture f(x, y);
  PDA(0).optimize(f.ctx, f.rng);

  auto const& actual = f.ctx.projector.value();
  auto const index   = f.ctx.pp_index_value.value();

  FeatureVector const expected = VEC(Feature, 0.0351398066F, -0.0574720800F, 0, 0, 0);

  ASSERT_COLLINEAR(expected, actual);
  ASSERT_GT(index, 0.99F) << "Optimal LDA projector for three groups has index near 1";
}

TEST(Projector, PDAOptimumProjectorLambdaOneHalfTwoGroups) {
  FeatureMatrix x = MAT(Feature, rows(4), 1, 0, 1, 1, 1, 4, 2, 1, 0, 0, 0, 4, 3, 0, 1, 1, 1, 1, 4, 0, 1, 2, 2, 1);

  GroupIdVector const y = VEC(GroupId, 0, 0, 1, 1);

  PDAFixture f(x, y);
  PDA(0.5).optimize(f.ctx, f.rng);

  auto const& actual = f.ctx.projector.value();
  auto const index   = f.ctx.pp_index_value.value();

  FeatureVector const expected = VEC(Feature, 0, 0, 0, 0, 0, 1);

  ASSERT_COLLINEAR(expected, actual);
  ASSERT_GT(index, 0.0F) << "PDA optimal projector has positive index";
}

TEST(Projector, PDAOptimumProjectorZeroColumn) {
  FeatureMatrix x =
      MAT(Feature, rows(4), 1, 0, 1, 1, 1, 4, 0, 2, 1, 0, 0, 0, 4, 0, 3, 0, 1, 1, 1, 1, 0, 4, 0, 1, 2, 2, 1, 0);

  GroupIdVector const y = VEC(GroupId, 0, 0, 1, 1);

  PDAFixture f(x, y);
  PDA(0.1).optimize(f.ctx, f.rng);

  auto const& actual = f.ctx.projector.value();
  auto const index   = f.ctx.pp_index_value.value();

  ASSERT_TRUE(actual.hasNaN()) << "Zero column with tiny sample produces degenerate (NaN) projector";
  ASSERT_TRUE(std::isnan(index)) << "Degenerate projector has NaN index";
}

TEST(Projector, PDALambdaOneBoundary) {
  FeatureMatrix x =
      MAT(Feature,
          rows(10),
          1,
          0,
          1,
          1,
          1,
          1,
          0,
          0,
          1,
          0,
          0,
          1,
          1,
          1,
          1,
          1,
          4,
          0,
          0,
          1,
          4,
          0,
          0,
          2,
          4,
          0,
          0,
          3,
          4,
          1,
          0,
          1,
          4,
          0,
          1,
          1,
          4,
          0,
          1,
          2);

  GroupIdVector const y = VEC(GroupId, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);

  PDAFixture f(x, y);
  PDA(1.0).optimize(f.ctx, f.rng);

  auto const& actual = f.ctx.projector.value();
  auto const index   = f.ctx.pp_index_value.value();

  // Lambda=1 means full penalization (diagonal covariance)
  // Projector should still point in the discriminating direction
  FeatureVector const expected = VEC(Feature, 1, 0, 0, 0);
  ASSERT_COLLINEAR(expected, actual);
  ASSERT_GT(index, 0.0F) << "PDA lambda=1 should still find a valid projector";
}


TEST(Projector, AssignsWeightToSelectedSignalColumnZeroesUnselected) {
  // var_selection = {0, 1}: PDA sees (col 0, col 1). Col 0 carries the
  // signal, col 1 is noise → reduced projector points along col 0;
  // expand routes it to full position 0 and zeros the unselected col 2.
  PDAFixture f(two_signal_x(), two_signal_y(), std::vector<int>{0, 1});
  PDA(0.5).optimize(f.ctx, f.rng);

  auto const& projector = f.ctx.projector.value();

  ASSERT_EQ(projector.size(), 3);
  EXPECT_NE(projector(0), 0.0F) << "col 0 is selected and carries the signal";
  EXPECT_FLOAT_EQ(projector(2), 0.0F) << "col 2 not in var_selection must expand to 0";
}

TEST(Projector, AssignsWeightToAlternateSelectedSignalColumn) {
  // Same data, var_selection = {1, 2}: PDA sees (col 1, col 2). Col 2
  // is the selected signal column; expand zeros the unselected col 0.
  PDAFixture f(two_signal_x(), two_signal_y(), std::vector<int>{1, 2});
  PDA(0.5).optimize(f.ctx, f.rng);

  auto const& projector = f.ctx.projector.value();

  ASSERT_EQ(projector.size(), 3);
  EXPECT_FLOAT_EQ(projector(0), 0.0F) << "col 0 not in var_selection must expand to 0";
  EXPECT_NE(projector(2), 0.0F) << "col 2 is selected and carries the signal";
}
