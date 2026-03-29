#include <gtest/gtest.h>
#include <Eigen/Dense>

#include "models/Tree.hpp"
#include "models/TreeCondition.hpp"
#include "models/TreeResponse.hpp"

#include "models/TrainingSpec.hpp"
#include "models/TrainingSpecPDA.hpp"
#include "models/TrainingSpecUPDA.hpp"

#include "stats/Simulation.hpp"
#include "stats/Stats.hpp"
#include "utils/Macros.hpp"

#include "serialization/Json.hpp"

using namespace ppforest2;
using namespace ppforest2::pp;
using namespace ppforest2::stats;
using namespace ppforest2::types;
using namespace ppforest2::math;
using namespace ppforest2::serialization;

static Projector as_projector(std::vector<Feature> vector) {
  Eigen::Map<Projector > projector(vector.data(), vector.size());
  return projector;
}

TEST(TreeResponse, EqualsEqualResponses) {
  TreeResponse r1(1);
  TreeResponse r2(1);

  ASSERT_TRUE(r1 == r2);
}

TEST(TreeResponse, EqualsDifferentResponses) {
  TreeResponse r1(1);
  TreeResponse r2(2);

  ASSERT_FALSE(r1 == r2);
}

TEST(TreeCondition, EqualsEqualConditions) {
  TreeCondition c1(
    as_projector({ 1.0, 2.0 }),
    3.0,
    TreeResponse::make(1),
    TreeResponse::make(2));

  TreeCondition c2(
    as_projector({ 1.0, 2.0 }),
    3.0,
    TreeResponse::make(1),
    TreeResponse::make(2));

  ASSERT_TRUE(c1 == c2);
}

TEST(TreeCondition, EqualsCollinearProjectors) {
  TreeCondition c1(
    as_projector({ 1.0, 1.0 }),
    3.0,
    TreeResponse::make(1),
    TreeResponse::make(2));

  TreeCondition c2(
    as_projector({ 2.0, 2.0 }),
    3.0,
    TreeResponse::make(1),
    TreeResponse::make(2));

  ASSERT_TRUE(c1 == c2);
}

TEST(TreeCondition, EqualsApproximateThresholds) {
  TreeCondition c1(
    as_projector({ 1.0, 2.0 }),
    3.0,
    TreeResponse::make(1),
    TreeResponse::make(2));

  TreeCondition c2(
    as_projector({ 1.0, 2.0 }),
    3.000000000000001,
    TreeResponse::make(1),
    TreeResponse::make(2));

  ASSERT_TRUE(c1 == c2);
}

TEST(TreeCondition, EqualsNonCollinearProjectors) {
  TreeCondition c1(
    as_projector({ 1.0, 0.0 }),
    3.0,
    TreeResponse::make(1),
    TreeResponse::make(2));

  TreeCondition c2(
    as_projector({ 0.0, 1.0 }),
    3.0,
    TreeResponse::make(1),
    TreeResponse::make(2));

  ASSERT_FALSE(c1 == c2);
}

TEST(TreeCondition, EqualsDifferentThresholds) {
  TreeCondition c1(
    as_projector({ 1.0, 2.0 }),
    3.0,
    TreeResponse::make(1),
    TreeResponse::make(2));

  TreeCondition c2(
    as_projector({ 1.0, 2.0 }),
    4.0,
    TreeResponse::make(1),
    TreeResponse::make(2));

  ASSERT_FALSE(c1 == c2);
}

TEST(TreeCondition, EqualsDifferentResponses) {
  TreeCondition c1(
    as_projector({ 1.0, 2.0 }),
    3.0,
    TreeResponse::make(1),
    TreeResponse::make(2));

  TreeCondition c2(
    as_projector({ 1.0, 2.0 }),
    3.0,
    TreeResponse::make(1),
    TreeResponse::make(3));

  ASSERT_FALSE(c1 == c2);
}

TEST(TreeCondition, EqualsDifferentStructures) {
  TreeCondition c1(
    as_projector({ 1.0, 2.0 }),
    3.0,
    TreeResponse::make(1),
    TreeResponse::make(2));

  TreeCondition c2(
    as_projector({ 1.0, 2.0 }),
    3.0,
    TreeResponse::make(1),
    TreeCondition::make(
      as_projector({ 1.0, 2.0 }),
      3.0,
      TreeResponse::make(1),
      TreeResponse::make(2)));

  ASSERT_FALSE(c1 == c2);
}

TEST(Tree, EqualsEqualTrees) {
  Tree t1(
    TreeCondition::make(
      as_projector({ 1.0, 2.0 }),
      3.0,
      TreeResponse::make(1),
      TreeCondition::make(
        as_projector({ 1.0, 2.0 }),
        3.0,
        TreeResponse::make(1),
        TreeResponse::make(2))));

  Tree t2(
    TreeCondition::make(
      as_projector({ 1.0, 2.0 }),
      3.0,
      TreeResponse::make(1),
      TreeCondition::make(
        as_projector({ 1.0, 2.0 }),
        3.0,
        TreeResponse::make(1),
        TreeResponse::make(2))));

  ASSERT_TRUE(t1 == t2);
}

TEST(Tree, EqualsDifferentTrees) {
  Tree t1(
    TreeCondition::make(
      as_projector({ 1.0, 2.0 }),
      3.0,
      TreeResponse::make(1),
      TreeResponse::make(2)));

  Tree t2(
    TreeCondition::make(
      as_projector({ 1.0, 2.0 }),
      3.0,
      TreeResponse::make(1),
      TreeResponse::make(3)));

  ASSERT_FALSE(t1 == t2);
}

TEST(Tree, TrainLDAUnivariateTwoGroups) {
  FeatureMatrix x = MAT(Feature, rows(10),
      1, 1, 1, 1, 1,
      2, 2, 2, 2, 2);

  ResponseVector y = VEC(Response,
      0, 0, 0, 0, 0,
      1, 1, 1, 1, 1);

  stats::RNG rng(0);

  Tree result = Tree::train(TrainingSpecPDA(0.0), x, y, rng);

  ResponseVector predictions = result.predict(x);

  ASSERT_EQ(error_rate(predictions, y), 0.0) << "Tree should achieve 0% training error on well-separated 2-group data";
}

TEST(Tree, TrainLDAUnivariateThreeGroups) {
  FeatureMatrix x = MAT(Feature, rows(15),
      1, 1, 1, 1, 1,
      2, 2, 2, 2, 2,
      3, 3, 3, 3, 3);

  ResponseVector y = VEC(Response,
      0, 0, 0, 0, 0,
      1, 1, 1, 1, 1,
      2, 2, 2, 2, 2);

  stats::RNG rng(0);

  Tree result = Tree::train(TrainingSpecPDA(0.0), x, y, rng);

  ResponseVector predictions = result.predict(x);

  ASSERT_EQ(error_rate(predictions, y), 0.0) << "Tree should achieve 0% training error on well-separated 3-group data";
}

TEST(Tree, TrainLDAMultivariateTwoGroups) {
  FeatureMatrix x = MAT(Feature, rows(10),
      1, 0, 1, 1,
      1, 1, 0, 0,
      1, 0, 0, 1,
      1, 1, 1, 1,
      4, 0, 0, 1,
      4, 0, 0, 2,
      4, 0, 0, 3,
      4, 1, 0, 1,
      4, 0, 1, 1,
      4, 0, 1, 2);

  ResponseVector y = VEC(Response,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1);

  stats::RNG rng(0);

  Tree result = Tree::train(TrainingSpecPDA(0.0), x, y, rng);

  ResponseVector predictions = result.predict(x);

  ASSERT_EQ(error_rate(predictions, y), 0.0) << "Tree should achieve 0% training error on well-separated 2-group data";
}

TEST(Tree, TrainLDAMultivariateThreeGroupsProperties) {
  FeatureMatrix x = MAT(Feature, rows(30),
      1, 0, 1, 1, 1,
      1, 0, 1, 0, 0,
      1, 0, 0, 0, 1,
      1, 0, 1, 2, 1,
      1, 0, 0, 1, 1,
      1, 1, 1, 1, 0,
      1, 0, 0, 2, 1,
      1, 0, 1, 1, 2,
      1, 0, 0, 2, 0,
      1, 0, 2, 1, 0,
      2, 5, 0, 0, 1,
      2, 5, 0, 0, 2,
      3, 5, 1, 0, 2,
      2, 5, 1, 0, 1,
      2, 5, 0, 1, 1,
      2, 5, 0, 1, 2,
      2, 5, 2, 1, 1,
      2, 5, 1, 1, 1,
      2, 5, 1, 1, 2,
      2, 5, 2, 1, 2,
      2, 5, 1, 2, 1,
      2, 5, 2, 1, 1,
      9, 8, 0, 0, 1,
      9, 8, 0, 0, 2,
      9, 8, 1, 0, 2,
      9, 8, 1, 0, 1,
      9, 8, 0, 1, 1,
      9, 8, 0, 1, 2,
      9, 8, 2, 1, 1,
      9, 8, 1, 1, 1);

  ResponseVector y = VEC(Response,
      0,
      0,
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
      2,
      2,
      2,
      2,
      2,
      2,
      2,
      2);

  stats::RNG rng(0);

  Tree result = Tree::train(TrainingSpecPDA(0.0), x, y, rng);

  // Property: tree should predict training data perfectly (well-separated groups)
  ResponseVector predictions = result.predict(x);

  ASSERT_EQ(error_rate(predictions, y), 0.0) << "Tree should achieve 0% training error on well-separated 3-group data";
}

TEST(Tree, TrainPDAMultivariateTwoGroupsProperties) {
  FeatureMatrix x = MAT(Feature, rows(10),
      1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      4, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2,
      5, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3,
      4, 0, 0, 3, 2, 2, 2, 2, 2, 2, 2, 2,
      4, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2,
      4, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
      4, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2);

  ResponseVector y = VEC(Response,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1);

  stats::RNG rng(0);

  Tree result = Tree::train(TrainingSpecPDA(0.5), x, y, rng);

  // Property: tree should predict training data perfectly
  ResponseVector predictions = result.predict(x);

  ASSERT_EQ(error_rate(predictions, y), 0.0) << "PDA tree should achieve 0% training error on well-separated 2-group data";
}

TEST(TreeSimulation, PerfectSeparationZeroError) {
  RNG rng(42);
  SimulationParams params;
  params.mean_separation = 200.0f;
  params.sd              = 1.0f;

  auto data = simulate(90, 4, 3, rng, params);

  RNG train_rng(42);
  Tree tree = Tree::train(TrainingSpecPDA(0.0), data.x, data.y, train_rng);

  ResponseVector predictions = tree.predict(data.x);
  ASSERT_EQ(error_rate(predictions, data.y), 0.0) << "Tree should achieve 0% error on perfectly separated data";
}

TEST(TreeSimulation, HighOverlapBoundedError) {
  RNG rng(42);
  SimulationParams params;
  params.mean_separation = 5.0f;
  params.sd              = 50.0f;

  auto data = simulate(200, 4, 3, rng, params);

  RNG train_rng(42);
  Tree tree = Tree::train(TrainingSpecPDA(0.0), data.x, data.y, train_rng);

  ResponseVector predictions = tree.predict(data.x);
  double err                 = error_rate(predictions, data.y);

  ASSERT_LT(err, 0.80) << "Tree error rate on highly overlapping data should be bounded";
}

TEST(TreeSimulation, ManyClasses) {
  RNG rng(42);
  SimulationParams params;
  params.mean_separation = 50.0f;
  params.sd              = 10.0f;

  auto data = simulate(300, 4, 10, rng, params);

  RNG train_rng(42);
  Tree tree = Tree::train(TrainingSpecPDA(0.0), data.x, data.y, train_rng);

  ResponseVector predictions = tree.predict(data.x);
  double err                 = error_rate(predictions, data.y);

  ASSERT_LT(err, 0.50)
    << "Tree should handle 10 groups with reasonable error";
}

TEST(TreeSimulation, HighDimensionality) {
  RNG rng(42);
  SimulationParams params;
  params.mean_separation = 50.0f;
  params.sd              = 10.0f;

  auto data = simulate(100, 50, 3, rng, params);

  RNG train_rng(42);
  Tree tree = Tree::train(TrainingSpecPDA(0.0), data.x, data.y, train_rng);

  ResponseVector predictions = tree.predict(data.x);
  double err                 = error_rate(predictions, data.y);

  ASSERT_LT(err, 0.20)
    << "Tree should handle high-dimensional data (p=50)";
}

TEST(TreeSimulation, SingleFeature) {
  RNG rng(42);
  SimulationParams params;
  params.mean_separation = 50.0f;
  params.sd              = 10.0f;

  auto data = simulate(100, 1, 2, rng, params);

  RNG train_rng(42);
  Tree tree = Tree::train(TrainingSpecPDA(0.0), data.x, data.y, train_rng);

  ResponseVector predictions = tree.predict(data.x);
  double err                 = error_rate(predictions, data.y);

  ASSERT_LT(err, 0.10)
    << "Tree should handle univariate data";
}

TEST(TreeSimulation, PDAOnOverlappingData) {
  RNG rng(42);
  SimulationParams params;
  params.mean_separation = 10.0f;
  params.sd              = 20.0f;

  auto data = simulate(200, 4, 3, rng, params);

  RNG train_rng(42);
  Tree tree = Tree::train(TrainingSpecPDA(0.5), data.x, data.y, train_rng);

  ResponseVector predictions = tree.predict(data.x);
  double err                 = error_rate(predictions, data.y);

  ASSERT_LT(err, 0.80)
    << "PDA tree should produce bounded error on noisy data";
}

TEST(TreeSimulation, Deterministic) {
  RNG rng(42);
  auto data = simulate(90, 4, 3, rng);

  RNG rng1(42);
  Tree t1 = Tree::train(TrainingSpecPDA(0.0), data.x, data.y, rng1);

  RNG rng2(42);
  Tree t2 = Tree::train(TrainingSpecPDA(0.0), data.x, data.y, rng2);

  ASSERT_EQ(t1, t2) << "Same seed should produce identical trees";
}

TEST(Tree, PredictDataColumnUnivariateTwoGroups) {
  Tree tree = Tree(
    TreeCondition::make(
      as_projector({ 1.0 }),
      1.5,
      TreeResponse::make(0),
      TreeResponse::make(1)));


  FeatureVector input = VEC(Feature, 1.0);
  ASSERT_EQ(tree.predict(input), 0);

  input = VEC(Feature, 2.0);
  ASSERT_EQ(tree.predict(input), 1);
}

TEST(Tree, PredictDataColumnUnivariateThreeGroups) {
  Tree tree = Tree(
    TreeCondition::make(
      as_projector({ 1.0 }),
      1.75,
      TreeResponse::make(0),
      TreeCondition::make(
        as_projector({ 1.0 }),
        2.5,
        TreeResponse::make(1),
        TreeResponse::make(2))));

  FeatureVector input = VEC(Feature, 1.0);
  ASSERT_EQ(tree.predict(input), 0);

  input = VEC(Feature, 2.0);
  ASSERT_EQ(tree.predict(input), 1);

  input = VEC(Feature, 3.0);
  ASSERT_EQ(tree.predict(input), 2);
}

TEST(Tree, PredictDataColumnMultivariateTwoGroups) {
  Tree tree = Tree(
    TreeCondition::make(
      as_projector({ 1.0, 0.0, 0.0, 0.0 }),
      2.5,
      TreeResponse::make(0),
      TreeResponse::make(1)));

  FeatureVector input = VEC(Feature, 1, 0, 1, 1);
  ASSERT_EQ(tree.predict(input), 0);

  input = VEC(Feature, 4, 0, 0, 1);
  ASSERT_EQ(tree.predict(input), 1);
}

TEST(Tree, PredictDataColumnMultivariateThreeGroups) {
  Tree tree = Tree(
    TreeCondition::make(
      as_projector({ 0.9805806756909201, -0.19611613513818427, 0.0, 0.0, 0.0 }),
      4.118438837901864,
      TreeCondition::make(
        as_projector({ 0.0, 1.0, 0.0, 0.0, 0.0 }),
        2.5,
        TreeResponse::make(0),
        TreeResponse::make(1)),
      TreeResponse::make(2)));

  FeatureVector input = VEC(Feature, 1, 0, 0, 1, 1);
  ASSERT_EQ(tree.predict(input), 0);

  input = VEC(Feature, 2, 5, 0, 0, 1);
  ASSERT_EQ(tree.predict(input), 1);

  input = VEC(Feature, 9, 8, 0, 0, 1);
  ASSERT_EQ(tree.predict(input), 2);
}

TEST(Tree, PredictDataUnivariateTwoGroups) {
  Tree tree = Tree(
    TreeCondition::make(
      as_projector({ 1.0 }),
      1.5,
      TreeResponse::make(0),
      TreeResponse::make(1)));

  FeatureMatrix input = MAT(Feature, rows(2), 1.0, 2.0);

  ResponseVector result = tree.predict(input);

  ResponseVector expected = VEC(Response, 0, 1);

  ASSERT_EQ(result, expected);
}

TEST(Tree, PredictDataUnivariateThreeGroups) {
  Tree tree = Tree(
    TreeCondition::make(
      as_projector({ 1.0 }),
      1.75,
      TreeResponse::make(0),
      TreeCondition::make(
        as_projector({ 1.0 }),
        2.5,
        TreeResponse::make(1),
        TreeResponse::make(2))));

  FeatureMatrix input = MAT(Feature, rows(3), 1.0, 2.0, 3.0);

  ResponseVector result = tree.predict(input);

  ResponseVector expected = VEC(Response, 0, 1, 2);

  ASSERT_EQ(result, expected);
}

TEST(Tree, PredictDataMultivariateTwoGroups) {
  Tree tree = Tree(
    TreeCondition::make(
      as_projector({ 1.0, 0.0, 0.0, 0.0 }),
      2.5,
      TreeResponse::make(0),
      TreeResponse::make(1)));

  FeatureMatrix input = MAT(Feature, rows(2),
      1, 0, 1, 1,
      4, 0, 0, 1);

  ResponseVector result = tree.predict(input);

  ResponseVector expected = VEC(Response, 0, 1);

  ASSERT_EQ(result, expected);
}

TEST(Tree, PredictDataMultivariateThreeGroups) {
  Tree tree = Tree(
    TreeCondition::make(
      as_projector({ 0.9805806756909201, -0.19611613513818427, 0.0, 0.0, 0.0 }),
      4.118438837901864,
      TreeCondition::make(
        as_projector({ 0.0, 1.0, 0.0, 0.0, 0.0 }),
        2.5,
        TreeResponse::make(0),
        TreeResponse::make(1)),
      TreeResponse::make(2)));

  FeatureMatrix input = MAT(Feature, rows(3),
      1, 0, 0, 1, 1,
      2, 5, 0, 0, 1,
      9, 8, 0, 0, 1);

  ResponseVector result = tree.predict(input);

  ResponseVector expected = VEC(Response, 0, 1, 2);

  ASSERT_EQ(result, expected);
}

TEST(Tree, PredictProportionsTwoGroups) {
  Tree tree = Tree(
    TreeCondition::make(
      as_projector({ 1.0 }),
      1.5,
      TreeResponse::make(0),
      TreeResponse::make(1),
      nullptr,
      { 0, 1 }));

  FeatureMatrix input = MAT(Feature, rows(2), 1.0, 2.0);

  FeatureMatrix result = tree.predict(input, Proportions{});

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
  Tree tree = Tree(
    TreeCondition::make(
      as_projector({ 1.0 }),
      1.75,
      TreeResponse::make(0),
      TreeCondition::make(
        as_projector({ 1.0 }),
        2.5,
        TreeResponse::make(1),
        TreeResponse::make(2),
        nullptr,
        { 1, 2 }),
      nullptr,
      { 0, 1, 2 }));

  FeatureMatrix input = MAT(Feature, rows(3), 1.0, 2.0, 3.0);

  FeatureMatrix result = tree.predict(input, Proportions{});

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
  Tree tree = Tree(
    TreeCondition::make(
      as_projector({ 1.0, 0.0, 0.0, 0.0 }),
      2.5,
      TreeResponse::make(0),
      TreeResponse::make(1),
      nullptr,
      { 0, 1 }));

  FeatureMatrix input = MAT(Feature, rows(4),
      1, 0, 1, 1,
      4, 0, 0, 1,
      2, 1, 0, 0,
      3, 0, 1, 0);

  FeatureMatrix result = tree.predict(input, Proportions{});

  ASSERT_EQ(result.rows(), 4);
  ASSERT_EQ(result.cols(), 2);

  for (int i = 0; i < result.rows(); ++i) {
    EXPECT_FLOAT_EQ(result.row(i).sum(), Feature(1))
      << "Row " << i << " should sum to 1.0";
  }
}

TEST(Tree, PredictProportionsMatchesPredictions) {
  Tree tree = Tree(
    TreeCondition::make(
      as_projector({ 0.9805806756909201, -0.19611613513818427, 0.0, 0.0, 0.0 }),
      4.118438837901864,
      TreeCondition::make(
        as_projector({ 0.0, 1.0, 0.0, 0.0, 0.0 }),
        2.5,
        TreeResponse::make(0),
        TreeResponse::make(1),
        nullptr,
        { 0, 1 }),
      TreeResponse::make(2),
      nullptr,
      { 0, 1, 2 }));

  FeatureMatrix input = MAT(Feature, rows(3),
      1, 0, 0, 1, 1,
      2, 5, 0, 0, 1,
      9, 8, 0, 0, 1);

  ResponseVector predictions = tree.predict(input);
  FeatureMatrix proportions  = tree.predict(input, Proportions{});

  ASSERT_EQ(proportions.rows(), 3);
  ASSERT_EQ(proportions.cols(), 3);

  // The column with 1.0 in each row must match the predicted group
  for (int i = 0; i < input.rows(); ++i) {
    EXPECT_FLOAT_EQ(proportions(i, predictions(i)), Feature(1)) << "Row " << i << ": proportions should have 1.0 in column for predicted group";
  }
}
