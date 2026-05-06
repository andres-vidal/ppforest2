#include <gtest/gtest.h>

#include "models/ClassificationForest.hpp"
#include "models/ClassificationTree.hpp"
#include "models/Forest.hpp"
#include "models/Evaluation.hpp"
#include "models/TreeBranch.hpp"
#include "models/TreeLeaf.hpp"

#include "models/TrainingSpec.hpp"
#include "test/TestSpec.hpp"

#include "stats/Simulation.hpp"
#include "stats/Metrics.hpp"
#include "stats/Stats.hpp"
#include "test/Projectors.hpp"
#include "utils/Macros.hpp"

using namespace ppforest2;
using namespace ppforest2::stats;
using namespace ppforest2::test;
using namespace ppforest2::types;
using namespace ppforest2::pp;
using namespace ppforest2::math;

TEST(Forest, TrainLDAAllVariablesProperties) {
  FeatureMatrix x =
      MAT(Feature,
          rows(30),
          1,
          0,
          1,
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
          2,
          1,
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
          1,
          0,
          0,
          2,
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
          5,
          0,
          0,
          1,
          2,
          5,
          0,
          0,
          2,
          3,
          5,
          1,
          0,
          2,
          2,
          5,
          1,
          0,
          1,
          2,
          5,
          0,
          1,
          1,
          2,
          5,
          0,
          1,
          2,
          2,
          5,
          2,
          1,
          1,
          2,
          5,
          1,
          1,
          1,
          2,
          5,
          1,
          1,
          2,
          2,
          5,
          2,
          1,
          2,
          2,
          5,
          1,
          2,
          1,
          2,
          5,
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

  OutcomeVector y =
      VEC(Outcome, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2);

  int const n_vars   = x.cols();
  float const lambda = 0;
  int const seed     = 0;

  auto result_ptr = Forest::train(
      TrainingSpec::builder(types::Mode::Classification)
          .size(4)
          .pp(pp::pda(lambda))
          .vars(vars::uniform(n_vars))
          .build(),
      x,
      y
  );
  Forest const& result = *result_ptr;

  ASSERT_EQ(result.trees.size(), 4);
  ASSERT_EQ(result.training_spec->seed, seed);

  OutcomeVector const predictions = result.predict(x);
  double const err                = error_rate(predictions, y);

  ASSERT_EQ(err, 0.0) << "Forest should achieve 0% training error on well-separated 3-group data";
}

TEST(Forest, TrainLDASomeVariablesProperties) {
  FeatureMatrix x =
      MAT(Feature,
          rows(30),
          1,
          0,
          1,
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
          2,
          1,
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
          1,
          0,
          0,
          2,
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
          5,
          0,
          0,
          1,
          2,
          5,
          0,
          0,
          2,
          3,
          5,
          1,
          0,
          2,
          2,
          5,
          1,
          0,
          1,
          2,
          5,
          0,
          1,
          1,
          2,
          5,
          0,
          1,
          2,
          2,
          5,
          2,
          1,
          1,
          2,
          5,
          1,
          1,
          1,
          2,
          5,
          1,
          1,
          2,
          2,
          5,
          2,
          1,
          2,
          2,
          5,
          1,
          2,
          1,
          2,
          5,
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

  OutcomeVector y =
      VEC(Outcome, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2);

  int const n_vars   = 2;
  float const lambda = 0;
  int const seed     = 1;


  auto result_ptr = Forest::train(
      TrainingSpec::builder(types::Mode::Classification)
          .size(4)
          .seed(seed)
          .pp(pp::pda(lambda))
          .vars(vars::uniform(n_vars))
          .build(),
      x,
      y
  );
  Forest const& result = *result_ptr;

  ASSERT_EQ(result.trees.size(), 4);
  ASSERT_EQ(result.training_spec->seed, seed);

  OutcomeVector const predictions = result.predict(x);
  double const err                = error_rate(predictions, y);

  ASSERT_LT(err, 0.30) << "Forest with subset of variables should still classify well-separated data";
}

TEST(Forest, TrainPDAAllVariablesProperties) {
  FeatureMatrix x =
      MAT(Feature,
          rows(10),
          1,
          0,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          0,
          0,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          0,
          0,
          1,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          0,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          1,
          4,
          0,
          0,
          1,
          2,
          2,
          2,
          2,
          2,
          2,
          2,
          2,
          5,
          0,
          0,
          3,
          3,
          3,
          3,
          3,
          3,
          3,
          3,
          3,
          4,
          0,
          0,
          3,
          2,
          2,
          2,
          2,
          2,
          2,
          2,
          2,
          4,
          1,
          0,
          1,
          2,
          2,
          2,
          2,
          2,
          2,
          2,
          2,
          4,
          0,
          1,
          1,
          2,
          2,
          2,
          2,
          2,
          2,
          2,
          2,
          4,
          0,
          1,
          2,
          2,
          2,
          2,
          2,
          2,
          2,
          2,
          2);

  OutcomeVector y = VEC(Outcome, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1);

  int const n_vars   = x.cols();
  float const lambda = 0.1;
  int const seed     = 0;

  auto result_ptr = Forest::train(
      TrainingSpec::builder(types::Mode::Classification)
          .size(4)
          .pp(pp::pda(lambda))
          .vars(vars::uniform(n_vars))
          .build(),
      x,
      y
  );
  Forest const& result = *result_ptr;

  ASSERT_EQ(result.trees.size(), 4);
  ASSERT_EQ(result.training_spec->seed, seed);

  OutcomeVector const predictions = result.predict(x);
  double const err                = error_rate(predictions, y);

  ASSERT_EQ(err, 0.0) << "PDA forest should achieve 0% training error on well-separated 2-group data";
}

TEST(ForestSimulation, PerfectSeparationLowOOBError) {
  RNG rng(0);
  simulation::params::Classification params;
  params.mean_separation = 200.0F;
  params.sd              = 1.0F;

  auto data = simulate(90, 4, 3, rng, params);

  auto forest = ClassificationForest::train(
      TrainingSpec::builder(types::Mode::Classification).size(20).threads(1).vars(vars::uniform(4)).build(),
      data.x,
      data.y
  );

  auto const err = oob_error(*forest, data.x, data.y);

  ASSERT_TRUE(err.has_value());
  ASSERT_GE(*err, 0.0);
  ASSERT_LE(*err, 0.05) << "Forest should achieve near-zero OOB error on perfectly separated data";
}

TEST(ForestSimulation, HighOverlapBoundedError) {
  RNG rng(0);
  simulation::params::Classification params;
  params.mean_separation = 5.0F;
  params.sd              = 50.0F;

  auto data = simulate(200, 4, 3, rng, params);

  auto forest = ClassificationForest::train(
      TrainingSpec::builder(types::Mode::Classification).size(20).threads(1).vars(vars::uniform(4)).build(),
      data.x,
      data.y
  );


  OutcomeVector const predictions = forest->predict(data.x);
  double const err                = error_rate(predictions, data.y);

  ASSERT_LT(err, 0.80) << "Forest error on highly overlapping data should be bounded";
}

TEST(ForestSimulation, ManyClasses) {
  RNG rng(0);
  simulation::params::Classification params;
  params.mean_separation = 50.0F;
  params.sd              = 10.0F;

  auto data = simulate(300, 4, 10, rng, params);

  auto forest = ClassificationForest::train(
      TrainingSpec::builder(types::Mode::Classification).size(20).threads(1).vars(vars::uniform(4)).build(),
      data.x,
      data.y
  );


  OutcomeVector const predictions = forest->predict(data.x);
  double const err                = error_rate(predictions, data.y);

  ASSERT_LT(err, 0.30) << "Forest should handle 10 groups with reasonable error";
}

TEST(ForestSimulation, HighDimensionality) {
  RNG rng(0);
  simulation::params::Classification params;
  params.mean_separation = 50.0F;
  params.sd              = 10.0F;

  auto data = simulate(100, 50, 3, rng, params);

  auto forest = ClassificationForest::train(
      TrainingSpec::builder(types::Mode::Classification).size(20).threads(1).vars(vars::uniform(7)).build(),
      data.x,
      data.y
  );


  OutcomeVector const predictions = forest->predict(data.x);
  double const err                = error_rate(predictions, data.y);

  ASSERT_LT(err, 0.20) << "Forest should handle high-dimensional data (p=50)";
}

TEST(ForestSimulation, Deterministic) {
  RNG rng(0);
  auto data = simulate(90, 4, 3, rng);

  auto f1_ptr = Forest::train(
      TrainingSpec::builder(types::Mode::Classification).size(10).threads(1).vars(vars::uniform(4)).build(),
      data.x,
      data.y
  );
  Forest const& f1 = *f1_ptr;
  auto f2_ptr      = Forest::train(
      TrainingSpec::builder(types::Mode::Classification).size(10).threads(1).vars(vars::uniform(4)).build(),
      data.x,
      data.y
  );
  Forest const& f2 = *f2_ptr;

  ASSERT_EQ(f1, f2) << "Same seed should produce identical forests";
}

TEST(ForestSimulation, PDAOnOverlappingData) {
  RNG rng(0);
  simulation::params::Classification params;
  params.mean_separation = 10.0F;
  params.sd              = 20.0F;

  auto data = simulate(200, 4, 3, rng, params);

  auto forest = ClassificationForest::train(
      TrainingSpec::builder(types::Mode::Classification)
          .size(20)
          .threads(1)
          .pp(pp::pda(0.5F))
          .vars(vars::uniform(4))
          .build(),
      data.x,
      data.y
  );


  OutcomeVector const predictions = forest->predict(data.x);
  double const err                = error_rate(predictions, data.y);

  ASSERT_LT(err, 0.80) << "PDA forest should produce bounded error on noisy data";
}

TEST(ForestSimulation, LargeDataset) {
  RNG rng(0);
  simulation::params::Classification params;
  params.mean_separation = 50.0F;
  params.sd              = 10.0F;

  auto data = simulate(2000, 10, 4, rng, params);

  auto forest = ClassificationForest::train(
      TrainingSpec::builder(types::Mode::Classification).size(10).threads(1).vars(vars::uniform(5)).build(),
      data.x,
      data.y
  );


  OutcomeVector const predictions = forest->predict(data.x);
  double const err                = error_rate(predictions, data.y);

  ASSERT_LT(err, 0.10) << "Forest should handle large datasets efficiently";
}

namespace {
  ClassificationForest build_three_group_forest() {
    ClassificationForest forest(test::classification_spec(), ClassificationForest::Groups{0, 1, 2});

    forest.add_tree(BaggedTree::make(
        std::make_unique<ClassificationTree>(
            TreeBranch::make(
                as_projector({0.0, 0.0, 0.0, 0.598, -0.801}),
                -0.348,
                TreeBranch::make(
                    as_projector({0.9995, 0.0, -0.031, 0.0, 0.0}), 5.553, TreeLeaf::make(1), TreeLeaf::make(2)
                ),
                TreeLeaf::make(0)
            ),
            test::classification_spec(),
            ClassificationTree::Groups{0, 1, 2}
        ),
        std::vector<int>{}
    ));

    forest.add_tree(BaggedTree::make(
        std::make_unique<ClassificationTree>(
            TreeBranch::make(
                as_projector({0.9998, 0.0, -0.019, 0.0, 0.0}),
                5.300,
                TreeBranch::make(
                    as_projector({0.999, 0.0, 0.0, 0.046, 0.0}), 1.609, TreeLeaf::make(0), TreeLeaf::make(1)
                ),
                TreeLeaf::make(2)
            ),
            test::classification_spec(),
            ClassificationTree::Groups{0, 1, 2}
        ),
        std::vector<int>{}
    ));

    forest.add_tree(BaggedTree::make(
        std::make_unique<ClassificationTree>(
            TreeBranch::make(
                as_projector({0.974, -0.226, 0.0, 0.0, 0.0}),
                3.955,
                TreeBranch::make(
                    as_projector({0.0, 0.9996, -0.030, 0.0, 0.0}), 2.622, TreeLeaf::make(0), TreeLeaf::make(1)
                ),
                TreeLeaf::make(2)
            ),
            test::classification_spec(),
            ClassificationTree::Groups{0, 1, 2}
        ),
        std::vector<int>{}
    ));

    forest.add_tree(BaggedTree::make(
        std::make_unique<ClassificationTree>(
            TreeBranch::make(
                as_projector({0.962, 0.0, 0.0, 0.0, -0.275}),
                4.735,
                TreeBranch::make(
                    as_projector({0.0, 0.0, 0.377, 0.0, -0.926}), -0.832, TreeLeaf::make(1), TreeLeaf::make(0)
                ),
                TreeLeaf::make(2)
            ),
            test::classification_spec(),
            ClassificationTree::Groups{0, 1, 2}
        ),
        std::vector<int>{}
    ));

    return forest;
  }
}

TEST(Forest, PredictSingleObservation) {
  ClassificationForest const forest = build_three_group_forest();

  ASSERT_EQ(0, forest.predict(VEC(Feature, 1, 0, 1, 1, 1)));
  ASSERT_EQ(1, forest.predict(VEC(Feature, 2, 5, 0, 0, 1)));
  ASSERT_EQ(2, forest.predict(VEC(Feature, 9, 8, 1, 1, 1)));
}

TEST(Forest, PredictBatch) {
  ClassificationForest const forest = build_three_group_forest();

  FeatureMatrix x =
      MAT(Feature, rows(6), 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 2, 5, 0, 0, 1, 2, 5, 1, 0, 1, 9, 8, 0, 0, 1, 9, 8, 1, 0, 2);

  OutcomeVector const result   = forest.predict(x);
  OutcomeVector const expected = VEC(Outcome, 0, 0, 1, 1, 2, 2);

  ASSERT_EQ(expected, result);
}


// ---------------------------------------------------------------------------
// Edge cases — "doesn't blow up" tests
// ---------------------------------------------------------------------------

TEST(ForestEdgeCase, ConstantFeatureColumn) {
  FeatureMatrix x = MAT(Feature, rows(6), 5, 1, 5, 2, 5, 3, 5, 7, 5, 8, 5, 9);
  OutcomeVector y = VEC(Outcome, 0, 0, 0, 1, 1, 1);

  auto forest =
      ClassificationForest::train(TrainingSpec::builder(types::Mode::Classification).size(5).threads(1).build(), x, y);

  OutcomeVector const predictions = forest->predict(x);

  ASSERT_EQ(predictions.size(), y.size());
  EXPECT_TRUE((predictions.array() >= 0).all() && (predictions.array() <= 1).all());
}

TEST(ForestEdgeCase, SingleObservationPerGroup) {
  FeatureMatrix x = MAT(Feature, rows(2), 1, 0, 0, 1);
  OutcomeVector y = VEC(Outcome, 0, 1);

  auto forest =
      ClassificationForest::train(TrainingSpec::builder(types::Mode::Classification).size(5).threads(1).build(), x, y);

  OutcomeVector const predictions = forest->predict(x);
  ASSERT_EQ(predictions.size(), y.size());
  EXPECT_TRUE((predictions.array() >= 0).all() && (predictions.array() <= 1).all());
}

TEST(ForestEdgeCase, MinimalDataset) {
  FeatureMatrix x = MAT(Feature, rows(2), 1, 9);
  OutcomeVector y = VEC(Outcome, 0, 1);

  auto forest =
      ClassificationForest::train(TrainingSpec::builder(types::Mode::Classification).size(5).threads(1).build(), x, y);

  OutcomeVector const predictions = forest->predict(x);
  ASSERT_EQ(predictions.size(), y.size());
  EXPECT_TRUE((predictions.array() >= 0).all() && (predictions.array() <= 1).all());
}

TEST(ForestEdgeCase, ExtremeImbalance) {
  // clang-format off
  FeatureMatrix x = MAT(Feature, rows(20),
    0, 0,  1, 1,  2, 2,  3, 0,  4, 1,
    0, 2,  1, 0,  2, 1,  3, 2,  4, 0,
    0, 1,  1, 2,  2, 0,  3, 1,  4, 2,
    0, 0,  1, 1,  2, 2,  90, 90,  91, 91);
  OutcomeVector y = VEC(Outcome,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1);
  // clang-format on

  auto forest =
      ClassificationForest::train(TrainingSpec::builder(types::Mode::Classification).size(10).threads(1).build(), x, y);


  OutcomeVector const predictions = forest->predict(x);
  ASSERT_EQ(predictions.size(), y.size());
  EXPECT_TRUE((predictions.array() >= 0).all() && (predictions.array() <= 1).all());
}
