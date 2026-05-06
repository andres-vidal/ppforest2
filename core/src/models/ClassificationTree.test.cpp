#include <gtest/gtest.h>
#include <Eigen/Dense>

#include "models/Tree.hpp"
#include "models/ClassificationTree.hpp"
#include "models/Model.hpp"
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
using namespace ppforest2::pp;
using namespace ppforest2::stats;
using namespace ppforest2::test;
using namespace ppforest2::types;
using namespace ppforest2::math;

TEST(Tree, EqualsEqualTrees) {
  ClassificationTree const t1(
      TreeBranch::make(
          as_projector({1.0, 2.0}),
          3.0,
          TreeLeaf::make(1),
          TreeBranch::make(as_projector({1.0, 2.0}), 3.0, TreeLeaf::make(1), TreeLeaf::make(2))
      ),
      test::classification_spec(),
      ClassificationTree::Groups{1, 2}
  );

  ClassificationTree const t2(
      TreeBranch::make(
          as_projector({1.0, 2.0}),
          3.0,
          TreeLeaf::make(1),
          TreeBranch::make(as_projector({1.0, 2.0}), 3.0, TreeLeaf::make(1), TreeLeaf::make(2))
      ),
      test::classification_spec(),
      ClassificationTree::Groups{1, 2}
  );

  ASSERT_TRUE(t1 == t2);
}

TEST(Tree, EqualsDifferentTrees) {
  ClassificationTree const t1(
      TreeBranch::make(as_projector({1.0, 2.0}), 3.0, TreeLeaf::make(1), TreeLeaf::make(2)),
      test::classification_spec(),
      ClassificationTree::Groups{1, 2}
  );

  ClassificationTree const t2(
      TreeBranch::make(as_projector({1.0, 2.0}), 3.0, TreeLeaf::make(1), TreeLeaf::make(3)),
      test::classification_spec(),
      ClassificationTree::Groups{1, 3}
  );

  ASSERT_FALSE(t1 == t2);
}

TEST(Tree, TrainLDAUnivariateTwoGroups) {
  FeatureMatrix x = MAT(Feature, rows(10), 1, 1, 1, 1, 1, 2, 2, 2, 2, 2);

  OutcomeVector y = VEC(Outcome, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1);

  stats::RNG rng(0);

  auto const result = Tree::train(TrainingSpec::builder(types::Mode::Classification).build(), x, y, rng);

  OutcomeVector const predictions = result->predict(x);

  ASSERT_EQ(error_rate(predictions, y), 0.0) << "Tree should achieve 0% training error on well-separated 2-group data";
}

TEST(Tree, TrainLDAUnivariateThreeGroups) {
  FeatureMatrix x = MAT(Feature, rows(15), 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3);

  OutcomeVector y = VEC(Outcome, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2);

  stats::RNG rng(0);

  auto const result = Tree::train(TrainingSpec::builder(types::Mode::Classification).build(), x, y, rng);

  OutcomeVector const predictions = result->predict(x);

  ASSERT_EQ(error_rate(predictions, y), 0.0) << "Tree should achieve 0% training error on well-separated 3-group data";
}

TEST(Tree, TrainLDAMultivariateTwoGroups) {
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

  stats::RNG rng(0);

  auto const result = Tree::train(TrainingSpec::builder(types::Mode::Classification).build(), x, y, rng);

  OutcomeVector const predictions = result->predict(x);

  ASSERT_EQ(error_rate(predictions, y), 0.0) << "Tree should achieve 0% training error on well-separated 2-group data";
}

TEST(Tree, TrainLDAMultivariateThreeGroupsProperties) {
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

  stats::RNG rng(0);

  auto const result = Tree::train(TrainingSpec::builder(types::Mode::Classification).build(), x, y, rng);

  // Property: tree should predict training data perfectly (well-separated groups)
  OutcomeVector const predictions = result->predict(x);

  ASSERT_EQ(error_rate(predictions, y), 0.0) << "Tree should achieve 0% training error on well-separated 3-group data";
}

TEST(Tree, TrainPDAMultivariateTwoGroupsProperties) {
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

  stats::RNG rng(0);

  auto const result =
      Tree::train(TrainingSpec::builder(types::Mode::Classification).pp(pp::pda(0.5F)).build(), x, y, rng);

  // Property: tree should predict training data perfectly
  OutcomeVector const predictions = result->predict(x);

  ASSERT_EQ(error_rate(predictions, y), 0.0)
      << "PDA tree should achieve 0% training error on well-separated 2-group data";
}

TEST(TreeSimulation, PerfectSeparationZeroError) {
  RNG rng(0);
  simulation::params::Classification params;
  params.mean_separation = 200.0F;
  params.sd              = 1.0F;

  auto data = simulate(90, 4, 3, rng, params);

  RNG train_rng(0);
  auto const tree = Tree::train(TrainingSpec::builder(types::Mode::Classification).build(), data.x, data.y, train_rng);

  OutcomeVector const predictions = tree->predict(data.x);
  ASSERT_EQ(error_rate(predictions, data.y), 0.0) << "Tree should achieve 0% error on perfectly separated data";
}

TEST(TreeSimulation, HighOverlapBoundedError) {
  RNG rng(0);
  simulation::params::Classification params;
  params.mean_separation = 5.0F;
  params.sd              = 50.0F;

  auto data = simulate(200, 4, 3, rng, params);

  RNG train_rng(0);
  auto const tree = Tree::train(TrainingSpec::builder(types::Mode::Classification).build(), data.x, data.y, train_rng);

  OutcomeVector const predictions = tree->predict(data.x);
  double const err                = error_rate(predictions, data.y);

  ASSERT_LT(err, 0.80) << "Tree error rate on highly overlapping data should be bounded";
}

TEST(TreeSimulation, ManyClasses) {
  RNG rng(0);
  simulation::params::Classification params;
  params.mean_separation = 50.0F;
  params.sd              = 10.0F;

  auto data = simulate(300, 4, 10, rng, params);

  RNG train_rng(0);
  auto const tree = Tree::train(TrainingSpec::builder(types::Mode::Classification).build(), data.x, data.y, train_rng);

  OutcomeVector const predictions = tree->predict(data.x);
  double const err                = error_rate(predictions, data.y);

  ASSERT_LT(err, 0.50) << "Tree should handle 10 groups with reasonable error";
}

TEST(TreeSimulation, HighDimensionality) {
  RNG rng(0);
  simulation::params::Classification params;
  params.mean_separation = 50.0F;
  params.sd              = 10.0F;

  auto data = simulate(100, 50, 3, rng, params);

  RNG train_rng(0);
  auto const tree = Tree::train(TrainingSpec::builder(types::Mode::Classification).build(), data.x, data.y, train_rng);

  OutcomeVector const predictions = tree->predict(data.x);
  double const err                = error_rate(predictions, data.y);

  ASSERT_LT(err, 0.20) << "Tree should handle high-dimensional data (p=50)";
}

TEST(TreeSimulation, SingleFeature) {
  RNG rng(0);
  simulation::params::Classification params;
  params.mean_separation = 50.0F;
  params.sd              = 10.0F;

  auto data = simulate(100, 1, 2, rng, params);

  RNG train_rng(0);
  auto const tree = Tree::train(TrainingSpec::builder(types::Mode::Classification).build(), data.x, data.y, train_rng);

  OutcomeVector const predictions = tree->predict(data.x);
  double const err                = error_rate(predictions, data.y);

  ASSERT_LT(err, 0.10) << "Tree should handle univariate data";
}

TEST(TreeSimulation, PDAOnOverlappingData) {
  RNG rng(0);
  simulation::params::Classification params;
  params.mean_separation = 10.0F;
  params.sd              = 20.0F;

  auto data = simulate(200, 4, 3, rng, params);

  RNG train_rng(0);
  auto const tree = Tree::train(
      TrainingSpec::builder(types::Mode::Classification).pp(pp::pda(0.5F)).build(), data.x, data.y, train_rng
  );

  OutcomeVector const predictions = tree->predict(data.x);
  double const err                = error_rate(predictions, data.y);

  ASSERT_LT(err, 0.80) << "PDA tree should produce bounded error on noisy data";
}

TEST(TreeSimulation, Deterministic) {
  RNG rng(0);
  auto data = simulate(90, 4, 3, rng);

  RNG rng1(0);
  auto const t1 = Tree::train(TrainingSpec::builder(types::Mode::Classification).build(), data.x, data.y, rng1);

  RNG rng2(0);
  auto const t2 = Tree::train(TrainingSpec::builder(types::Mode::Classification).build(), data.x, data.y, rng2);

  ASSERT_EQ(*t1, *t2) << "Same seed should produce identical trees";
}

TEST(Tree, PredictDataColumnUnivariateTwoGroups) {
  ClassificationTree const tree(
      TreeBranch::make(as_projector({1.0}), 1.5, TreeLeaf::make(0), TreeLeaf::make(1)),
      test::classification_spec(),
      ClassificationTree::Groups{0, 1}
  );


  FeatureVector input = VEC(Feature, 1.0);
  ASSERT_EQ(tree.predict(input), 0);

  input = VEC(Feature, 2.0);
  ASSERT_EQ(tree.predict(input), 1);
}

TEST(Tree, PredictDataColumnUnivariateThreeGroups) {
  ClassificationTree const tree(
      TreeBranch::make(
          as_projector({1.0}),
          1.75,
          TreeLeaf::make(0),
          TreeBranch::make(as_projector({1.0}), 2.5, TreeLeaf::make(1), TreeLeaf::make(2))
      ),
      test::classification_spec(),
      ClassificationTree::Groups{0, 1, 2}
  );

  FeatureVector input = VEC(Feature, 1.0);
  ASSERT_EQ(tree.predict(input), 0);

  input = VEC(Feature, 2.0);
  ASSERT_EQ(tree.predict(input), 1);

  input = VEC(Feature, 3.0);
  ASSERT_EQ(tree.predict(input), 2);
}

TEST(Tree, PredictDataColumnMultivariateTwoGroups) {
  ClassificationTree const tree = ClassificationTree(
      TreeBranch::make(as_projector({1.0, 0.0, 0.0, 0.0}), 2.5, TreeLeaf::make(0), TreeLeaf::make(1)),
      test::classification_spec(),
      ClassificationTree::Groups{0, 1}
  );

  FeatureVector input = VEC(Feature, 1, 0, 1, 1);
  ASSERT_EQ(tree.predict(input), 0);

  input = VEC(Feature, 4, 0, 0, 1);
  ASSERT_EQ(tree.predict(input), 1);
}

TEST(Tree, PredictDataColumnMultivariateThreeGroups) {
  ClassificationTree const tree(
      TreeBranch::make(
          as_projector({0.9805806756909201, -0.19611613513818427, 0.0, 0.0, 0.0}),
          4.118438837901864,
          TreeBranch::make(as_projector({0.0, 1.0, 0.0, 0.0, 0.0}), 2.5, TreeLeaf::make(0), TreeLeaf::make(1)),
          TreeLeaf::make(2)
      ),
      test::classification_spec(),
      ClassificationTree::Groups{0, 1, 2}
  );

  FeatureVector input = VEC(Feature, 1, 0, 0, 1, 1);
  ASSERT_EQ(tree.predict(input), 0);

  input = VEC(Feature, 2, 5, 0, 0, 1);
  ASSERT_EQ(tree.predict(input), 1);

  input = VEC(Feature, 9, 8, 0, 0, 1);
  ASSERT_EQ(tree.predict(input), 2);
}

TEST(Tree, PredictDataUnivariateTwoGroups) {
  ClassificationTree const tree(
      TreeBranch::make(as_projector({1.0}), 1.5, TreeLeaf::make(0), TreeLeaf::make(1)),
      test::classification_spec(),
      ClassificationTree::Groups{0, 1}
  );

  FeatureMatrix const input = MAT(Feature, rows(2), 1.0, 2.0);

  OutcomeVector const result   = tree.predict(input);
  OutcomeVector const expected = VEC(Outcome, 0, 1);

  ASSERT_EQ(result, expected);
}

TEST(Tree, PredictDataUnivariateThreeGroups) {
  ClassificationTree const tree(
      TreeBranch::make(
          as_projector({1.0}),
          1.75,
          TreeLeaf::make(0),
          TreeBranch::make(as_projector({1.0}), 2.5, TreeLeaf::make(1), TreeLeaf::make(2))
      ),
      test::classification_spec(),
      ClassificationTree::Groups{0, 1, 2}
  );

  FeatureMatrix const input = MAT(Feature, rows(3), 1.0, 2.0, 3.0);

  OutcomeVector const result   = tree.predict(input);
  OutcomeVector const expected = VEC(Outcome, 0, 1, 2);

  ASSERT_EQ(result, expected);
}

TEST(Tree, PredictDataMultivariateTwoGroups) {
  ClassificationTree const tree = ClassificationTree(
      TreeBranch::make(as_projector({1.0, 0.0, 0.0, 0.0}), 2.5, TreeLeaf::make(0), TreeLeaf::make(1)),
      test::classification_spec(),
      ClassificationTree::Groups{0, 1}
  );

  FeatureMatrix const input = MAT(Feature, rows(2), 1, 0, 1, 1, 4, 0, 0, 1);

  OutcomeVector const result   = tree.predict(input);
  OutcomeVector const expected = VEC(Outcome, 0, 1);

  ASSERT_EQ(result, expected);
}

TEST(Tree, PredictDataMultivariateThreeGroups) {
  ClassificationTree tree(
      TreeBranch::make(
          as_projector({0.9805806756909201, -0.19611613513818427, 0.0, 0.0, 0.0}),
          4.118438837901864,
          TreeBranch::make(as_projector({0.0, 1.0, 0.0, 0.0, 0.0}), 2.5, TreeLeaf::make(0), TreeLeaf::make(1)),
          TreeLeaf::make(2)
      ),
      test::classification_spec(),
      ClassificationTree::Groups{0, 1, 2}
  );

  FeatureMatrix const input = MAT(Feature, rows(3), 1, 0, 0, 1, 1, 2, 5, 0, 0, 1, 9, 8, 0, 0, 1);

  OutcomeVector const result   = tree.predict(input);
  OutcomeVector const expected = VEC(Outcome, 0, 1, 2);

  ASSERT_EQ(result, expected) << "Tree should predict the correct groups for the input data";
}

TEST(Tree, PredictProportionsTwoGroups) {
  ClassificationTree const tree = ClassificationTree(
      TreeBranch::make(as_projector({1.0}), 1.5, TreeLeaf::make(0), TreeLeaf::make(1), {0, 1}),
      test::classification_spec(),
      ClassificationTree::Groups{0, 1}
  );

  FeatureMatrix const input = MAT(Feature, rows(2), 1.0, 2.0);

  FeatureMatrix const result = tree.predict(input, Proportions{});

  ASSERT_EQ(result.rows(), 2);
  ASSERT_EQ(result.cols(), 2);

  // Row 0 predicts group 0 → [1, 0]
  EXPECT_FLOAT_EQ(result(0, 0), 1.0);
  EXPECT_FLOAT_EQ(result(0, 1), 0.0);

  // Row 1 predicts group 1 → [0, 1]
  EXPECT_FLOAT_EQ(result(1, 0), 0.0);
  EXPECT_FLOAT_EQ(result(1, 1), 1.0);
}

TEST(Tree, PredictProportionsThreeGroups) {
  ClassificationTree const tree(
      TreeBranch::make(
          as_projector({1.0}),
          1.75,
          TreeLeaf::make(0),
          TreeBranch::make(as_projector({1.0}), 2.5, TreeLeaf::make(1), TreeLeaf::make(2), {1, 2}),
          {0, 1, 2}
      ),
      test::classification_spec(),
      ClassificationTree::Groups{0, 1, 2}
  );

  FeatureMatrix const input = MAT(Feature, rows(3), 1.0, 2.0, 3.0);

  FeatureMatrix const result = tree.predict(input, Proportions{});

  ASSERT_EQ(result.rows(), 3);
  ASSERT_EQ(result.cols(), 3);

  // Each row is one-hot for the predicted group
  EXPECT_FLOAT_EQ(result(0, 0), 1.0);
  EXPECT_FLOAT_EQ(result(0, 1), 0.0);
  EXPECT_FLOAT_EQ(result(0, 2), 0.0);

  EXPECT_FLOAT_EQ(result(1, 0), 0.0);
  EXPECT_FLOAT_EQ(result(1, 1), 1.0);
  EXPECT_FLOAT_EQ(result(1, 2), 0.0);

  EXPECT_FLOAT_EQ(result(2, 0), 0.0);
  EXPECT_FLOAT_EQ(result(2, 1), 0.0);
  EXPECT_FLOAT_EQ(result(2, 2), 1.0);
}

TEST(Tree, PredictProportionsRowsSumToOne) {
  ClassificationTree const tree(
      TreeBranch::make(as_projector({1.0, 0.0, 0.0, 0.0}), 2.5, TreeLeaf::make(0), TreeLeaf::make(1), {0, 1}),
      test::classification_spec(),
      ClassificationTree::Groups{0, 1}
  );

  FeatureMatrix const input = MAT(Feature, rows(4), 1, 0, 1, 1, 4, 0, 0, 1, 2, 1, 0, 0, 3, 0, 1, 0);

  FeatureMatrix const result = tree.predict(input, Proportions{});

  ASSERT_EQ(result.rows(), 4);
  ASSERT_EQ(result.cols(), 2);

  for (int i = 0; i < result.rows(); ++i) {
    EXPECT_FLOAT_EQ(result.row(i).sum(), Feature(1)) << "Row " << i << " should sum to 1.0";
  }
}

TEST(Tree, PredictProportionsMatchesPredictions) {
  ClassificationTree const tree(
      TreeBranch::make(
          as_projector({0.9805806756909201, -0.19611613513818427, 0.0, 0.0, 0.0}),
          4.118438837901864,
          TreeBranch::make(as_projector({0.0, 1.0, 0.0, 0.0, 0.0}), 2.5, TreeLeaf::make(0), TreeLeaf::make(1), {0, 1}),
          TreeLeaf::make(2),
          {0, 1, 2}
      ),
      test::classification_spec(),
      ClassificationTree::Groups{0, 1, 2}
  );

  FeatureMatrix const input = MAT(Feature, rows(3), 1, 0, 0, 1, 1, 2, 5, 0, 0, 1, 9, 8, 0, 0, 1);

  OutcomeVector predictions = tree.predict(input);
  FeatureMatrix proportions = tree.predict(input, Proportions{});

  ASSERT_EQ(proportions.rows(), 3);
  ASSERT_EQ(proportions.cols(), 3);

  // The column with 1.0 in each row must match the predicted group
  for (int i = 0; i < input.rows(); ++i) {
    EXPECT_FLOAT_EQ(proportions(i, static_cast<int>(predictions(i))), Feature(1))
        << "Row " << i << ": proportions should have 1.0 in column for predicted group";
  }
}

TEST(Tree, PredictProportionsHelperMatchesDirectCall) {
  ClassificationTree const tree(TreeLeaf::make(0), test::classification_spec(), ClassificationTree::Groups{0});

  FeatureMatrix x(3, 2);
  x << 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F;

  FeatureMatrix const via_helper = predict_proportions(tree, x);
  FeatureMatrix const via_direct = tree.predict(x, Proportions{});

  ASSERT_EQ_DATA(via_helper, via_direct);
}

// ---------------------------------------------------------------------------
// Edge cases — "doesn't blow up" tests
// ---------------------------------------------------------------------------

TEST(TreeEdgeCase, ConstantFeatureColumn) {
  FeatureMatrix x = MAT(Feature, rows(6), 5, 1, 5, 2, 5, 3, 5, 7, 5, 8, 5, 9);
  OutcomeVector y = VEC(Outcome, 0, 0, 0, 1, 1, 1);

  stats::RNG rng(0);
  auto const tree = Tree::train(TrainingSpec::builder(types::Mode::Classification).build(), x, y, rng);

  ASSERT_EQ_DATA(tree->predict(x), VEC(Outcome, 0, 0, 0, 0, 0, 0));
}

TEST(TreeEdgeCase, SingleObservationPerGroup) {
  FeatureMatrix x = MAT(Feature, rows(2), 1, 0, 0, 1);
  OutcomeVector y = VEC(Outcome, 0, 1);

  stats::RNG rng(0);
  auto const tree = Tree::train(TrainingSpec::builder(types::Mode::Classification).build(), x, y, rng);

  ASSERT_EQ_DATA(tree->predict(x), VEC(Outcome, 0, 1));
}

TEST(TreeEdgeCase, MinimalDataset) {
  FeatureMatrix x = MAT(Feature, rows(2), 1, 9);
  OutcomeVector y = VEC(Outcome, 0, 1);

  stats::RNG rng(0);
  auto const tree = Tree::train(TrainingSpec::builder(types::Mode::Classification).build(), x, y, rng);

  ASSERT_EQ_DATA(tree->predict(x), VEC(Outcome, 0, 1));
}

TEST(TreeEdgeCase, ExtremeImbalance) {
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

  stats::RNG rng(0);
  auto const tree = Tree::train(TrainingSpec::builder(types::Mode::Classification).build(), x, y, rng);

  ASSERT_EQ_DATA(tree->predict(x), y.cast<Outcome>());
}

namespace {
  // Visitor that collects all pp_index_value fields from a tree.
  struct IndexCollector : public TreeNode::Visitor {
    std::vector<Feature> values;

    void visit(TreeBranch const& node) override {
      values.push_back(node.pp_index_value);
      node.lower->accept(*this);
      node.upper->accept(*this);
    }

    void visit(TreeLeaf const& /*node*/) override {}
  };
}

TEST(PPIndexValue, StoredAfterTraining) {
  // Two well-separated groups: col 0 discriminates.
  FeatureMatrix x = MAT(Feature, rows(6), 0.0F, 0.5F, 0.1F, 0.3F, 0.2F, 0.7F, 9.8F, 0.4F, 9.9F, 0.6F, 9.7F, 0.2F);

  OutcomeVector y = VEC(Outcome, 0, 0, 0, 1, 1, 1);

  stats::RNG rng(0, 0);
  auto const tree = Tree::train(TrainingSpec::builder(types::Mode::Classification).build(), x, y, rng);

  IndexCollector collector;
  tree->root->accept(collector);

  ASSERT_FALSE(collector.values.empty());

  bool any_nonzero = false;
  for (Feature const v : collector.values) {
    if (v != Feature(0)) {
      any_nonzero = true;
      break;
    }
  }

  ASSERT_TRUE(any_nonzero) << "Expected at least one non-zero pp_index_value after training";
}

// ---------------------------------------------------------------------------
// Stop-rule integration: classification analog of `RegressionTree.MinSizeStop` /
// `RegressionTree.MinVarianceStop`. Verifies that a specific stop rule
// actually fires by measuring the resulting tree's depth — not just that
// training succeeds. `MaxDepth` is the mode-agnostic rule with a property
// that's directly observable from the tree structure.
// ---------------------------------------------------------------------------

namespace {
  // Recursive depth measure: 0 for a leaf, 1 + max(child depths) for a branch.
  struct DepthVisitor : public TreeNode::Visitor {
    int depth = 0;

    void visit(TreeLeaf const& /*node*/) override { depth = 0; }

    void visit(TreeBranch const& node) override {
      DepthVisitor left;
      node.lower->accept(left);
      DepthVisitor right;
      node.upper->accept(right);
      depth = 1 + std::max(left.depth, right.depth);
    }
  };

  int tree_depth(Tree const& tree) {
    DepthVisitor v;
    tree.root->accept(v);
    return v.depth;
  }
}

TEST(ClassificationTreeStop, MaxDepthZeroProducesRootLeaf) {
  // `max_depth(0)` allows only a root-level stump — the root must be a
  // leaf, not a branch. Two well-separated groups would otherwise split
  // immediately; this confirms the stop rule overrides that.
  FeatureMatrix x = MAT(Feature, rows(6), 0.0F, 0.5F, 0.1F, 0.3F, 0.2F, 0.7F, 9.8F, 0.4F, 9.9F, 0.6F, 9.7F, 0.2F);
  OutcomeVector y = VEC(Outcome, 0, 0, 0, 1, 1, 1);

  stats::RNG rng(0, 0);
  auto const tree =
      Tree::train(TrainingSpec::builder(types::Mode::Classification).stop(stop::max_depth(0)).build(), x, y, rng);

  EXPECT_EQ(tree_depth(*tree), 0) << "max_depth(0) must produce a single-leaf tree";
}

TEST(ClassificationTreeStop, MaxDepthBoundsTreeDepth) {
  // For a non-trivial cap, no path in the resulting tree may exceed
  // `max_depth`. We use three well-separated groups so the tree
  // *would* go deeper than 1 without the cap.
  FeatureMatrix x =
      MAT(Feature,
          rows(9),
          0.0F,
          0.5F,
          0.1F,
          0.3F,
          0.2F,
          0.7F,
          5.0F,
          0.4F,
          5.1F,
          0.6F,
          5.2F,
          0.2F,
          9.8F,
          0.4F,
          9.9F,
          0.6F,
          9.7F,
          0.2F);
  OutcomeVector y = VEC(Outcome, 0, 0, 0, 1, 1, 1, 2, 2, 2);

  stats::RNG rng(0, 0);
  auto const tree =
      Tree::train(TrainingSpec::builder(types::Mode::Classification).stop(stop::max_depth(1)).build(), x, y, rng);

  EXPECT_LE(tree_depth(*tree), 1) << "max_depth(1) must bound tree depth at 1";
}
