#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>

#include "models/Bagged.hpp"
#include "models/ClassificationForest.hpp"
#include "models/ClassificationTree.hpp"
#include "models/Forest.hpp"
#include "models/Evaluation.hpp"
#include "models/RegressionForest.hpp"
#include "models/RegressionTree.hpp"
#include "models/Tree.hpp"
#include "models/TrainingSpec.hpp"
#include "models/TreeBranch.hpp"
#include "models/TreeLeaf.hpp"
#include "stats/Simulation.hpp"
#include "test/Projectors.hpp"
#include "test/TestSpec.hpp"
#include "utils/Macros.hpp"

#include <cmath>

using namespace ppforest2;
using namespace ppforest2::pp;
using namespace ppforest2::stats;
using namespace ppforest2::test;
using namespace ppforest2::types;

namespace {
  // Two clusters — col 0 separates them (≈0 vs ≈9), col 1 is noise.
  // First 10 rows are group 0, last 10 are group 1.
  std::pair<FeatureMatrix, OutcomeVector> perfect_separation_data() {
    FeatureMatrix x =
        MAT(Feature,
            rows(20),
            0.0F,
            1.0F,
            0.1F,
            2.0F,
            0.2F,
            0.5F,
            0.3F,
            1.5F,
            0.4F,
            0.8F,
            0.5F,
            1.2F,
            0.6F,
            0.9F,
            0.7F,
            1.8F,
            0.8F,
            0.6F,
            0.9F,
            1.1F,
            9.0F,
            1.0F,
            9.1F,
            2.0F,
            9.2F,
            0.5F,
            9.3F,
            1.5F,
            9.4F,
            0.8F,
            9.5F,
            1.2F,
            9.6F,
            0.9F,
            9.7F,
            1.8F,
            9.8F,
            0.6F,
            9.9F,
            1.1F);
    OutcomeVector y = VEC(Outcome, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
    return {x, y};
  }

  // Hand-built classification forest with one tree: split on x[0] at 5.0,
  // left → group 0, right → group 1. Tests vary only the bootstrap sample —
  // and thus the OOB set — so it's the single parameter.
  ClassificationForest single_split_forest(std::vector<int> sample_indices) {
    ClassificationForest forest(test::classification_spec(), ClassificationForest::Groups{0, 1});
    forest.add_tree(BaggedTree::make(
        std::make_unique<ClassificationTree>(
            TreeBranch::make(as_projector({1.0F, 0.0F}), 5.0F, TreeLeaf::make(0), TreeLeaf::make(1), {0, 1}, 0.9F),
            TrainingSpec::builder(types::Mode::Classification).make(),
            ClassificationTree::Groups{0, 1}
        ),
        std::move(sample_indices)
    ));
    return forest;
  }

  // Hand-built regression forest. Each entry in `bags` becomes a one-leaf
  // tree whose prediction for any input is `leaf_value`. Tests vary the
  // per-tree bootstrap sample (and thus the OOB set) plus the leaf value
  // (and thus what predictions look like on OOB rows).
  RegressionForest leaf_regression_forest(std::vector<std::vector<int>> bags, Feature leaf_value = 0.0F) {
    RegressionForest forest(test::regression_spec());
    for (auto& sample : bags) {
      forest.add_tree(BaggedTree::make(
          std::make_unique<RegressionTree>(TreeLeaf::make(leaf_value), test::regression_spec()), std::move(sample)
      ));
    }
    return forest;
  }
}

// ---------------------------------------------------------------------------
// ViProjections — Tree overload (single tree, no forest wrapper)
// ---------------------------------------------------------------------------

TEST(ViProjections, TreeOverloadSingleNode) {
  // projector=[1,0], G_s=2, pp_index_value=0.9
  // VI2[0] = |1| / 2 = 0.5
  // VI2[1] = |0| / 2 = 0.0

  ClassificationTree const tree(
      TreeBranch::make(as_projector({1.0F, 0.0F}), 5.0F, TreeLeaf::make(0), TreeLeaf::make(1), {0, 1}, 0.9F),
      test::classification_spec(),
      ClassificationTree::Groups{0, 1}
  );

  FeatureVector vi2 = vi_projections(tree, 2);

  ASSERT_NEAR(vi2(0), 0.5F, 1e-5F);
  ASSERT_NEAR(vi2(1), 0.0F, 1e-5F);
}

TEST(ViProjections, TreeOverloadTwoNodes) {
  // Root:  projector=[1,0], G_s=3
  //   -> lower leaf
  //   -> inner: projector=[0,1], G_s=2
  //        -> leaf, leaf
  //
  // VI2[0] = |1|/3 + |0|/2 = 1/3
  // VI2[1] = |0|/3 + |1|/2 = 1/2


  ClassificationTree const tree(
      TreeBranch::make(
          as_projector({1.0F, 0.0F}),
          5.0F,
          TreeLeaf::make(0),
          TreeBranch::make(as_projector({0.0F, 1.0F}), 3.0F, TreeLeaf::make(1), TreeLeaf::make(2), {1, 2}, 0.6F),
          {0, 1, 2},
          0.8F
      ),
      test::classification_spec(),
      ClassificationTree::Groups{0, 1, 2}
  );

  FeatureVector vi2 = vi_projections(tree, 2);

  ASSERT_NEAR(vi2(0), 1.0F / 3.0F, 1e-5F);
  ASSERT_NEAR(vi2(1), 1.0F / 2.0F, 1e-5F);
}

TEST(ViProjections, TreeOverloadWithScale) {
  // projector=[0.5, 0.2], G_s=2
  // scale=[2.0, 3.0]
  // VI2[0] = |0.5| * 2.0 / 2 = 0.5
  // VI2[1] = |0.2| * 3.0 / 2 = 0.3


  ClassificationTree const tree(
      TreeBranch::make(as_projector({0.5F, 0.2F}), 5.0F, TreeLeaf::make(0), TreeLeaf::make(1), {0, 1}, 0.9F),
      test::classification_spec(),
      ClassificationTree::Groups{0, 1}
  );

  FeatureVector scale(2);
  scale << 2.0F, 3.0F;

  FeatureVector vi2 = vi_projections(tree, 2, &scale);

  ASSERT_NEAR(vi2(0), 0.5F, 1e-5F);
  ASSERT_NEAR(vi2(1), 0.3F, 1e-5F);
}

// ---------------------------------------------------------------------------
// VI2 — hand-built single-node tree (forest wrapper)
// ---------------------------------------------------------------------------

TEST(ViProjections, HandBuiltSingleNodeTree) {
  // Single tree: projector = [1, 0], pp_index_value = 0.9, G_s = 2.
  // Expected VI2[j] = |a_j| / G_s:
  //   VI2[0] = 1.0 / 2 = 0.5
  //   VI2[1] = 0.0 / 2 = 0.0
  ClassificationForest forest = single_split_forest({0, 1, 2, 3});

  FeatureVector vi2 = vi_projections(forest, 2);

  ASSERT_NEAR(vi2(0), 0.5F, 1e-5F);
  ASSERT_NEAR(vi2(1), 0.0F, 1e-5F);
}

TEST(ViProjections, RegressionHandBuiltSingleNodeTree) {
  // Regression analogue of HandBuiltSingleNodeTree. Regression branches
  // carry an artificial 2-group set {0, 1} (the "lower" and "upper"
  // virtual groups created by ByCutpoint), so VI2[j] = |a_j| / 2.
  // Single split on x[0] at 5.0: projector=[1,0], G_s=2.
  //   VI2[0] = 1.0 / 2 = 0.5
  //   VI2[1] = 0.0 / 2 = 0.0
  RegressionForest forest(test::regression_spec());
  forest.add_tree(BaggedTree::make(
      std::make_unique<RegressionTree>(
          TreeBranch::make(as_projector({1.0F, 0.0F}), 5.0F, TreeLeaf::make(0.0F), TreeLeaf::make(1.0F), {0, 1}, 0.9F),
          test::regression_spec()
      ),
      std::vector<int>{0, 1, 2, 3}
  ));

  FeatureVector vi2 = vi_projections(forest, 2);

  ASSERT_NEAR(vi2(0), 0.5F, 1e-5F);
  ASSERT_NEAR(vi2(1), 0.0F, 1e-5F);
}

TEST(ViProjections, HandBuiltTwoNodeTree) {
  // Root:  projector=[1,0], G_s=3, pp_index_value=0.8
  //   -> lower leaf
  //   -> inner: projector=[0,1], G_s=2, pp_index_value=0.6
  //        -> leaf, leaf
  //
  // VI2[0] = (1/B) * (|1|/3 + |0|/2) = 1/3
  // VI2[1] = (1/B) * (|0|/3 + |1|/2) = 1/2
  ClassificationForest forest(test::classification_spec(), ClassificationForest::Groups{0, 1, 2});
  forest.add_tree(BaggedTree::make(
      std::make_unique<ClassificationTree>(
          TreeBranch::make(
              as_projector({1.0F, 0.0F}),
              5.0F,
              TreeLeaf::make(0),
              TreeBranch::make(as_projector({0.0F, 1.0F}), 3.0F, TreeLeaf::make(1), TreeLeaf::make(2), {1, 2}, 0.6F),
              {0, 1, 2},
              0.8F
          ),
          TrainingSpec::builder(types::Mode::Classification).make(),
          ClassificationTree::Groups{0, 1, 2}
      ),
      std::vector<int>{0, 1, 2, 3}
  ));

  FeatureVector vi2 = vi_projections(forest, 2);

  ASSERT_NEAR(vi2(0), 1.0F / 3.0F, 1e-5F);
  ASSERT_NEAR(vi2(1), 1.0F / 2.0F, 1e-5F);
}

// ---------------------------------------------------------------------------
// ViWeightedProjections — hand-built single-node tree
// ---------------------------------------------------------------------------

TEST(ViWeightedProjections, HandBuiltSingleNodeTree) {
  // All 4 rows are in-bag, so the single tree has empty OOB. A tree that
  // saw every training row has no out-of-sample view to evaluate against,
  // so it can't contribute to the OOB-weighted VI3 average and is dropped.
  // With only one tree in the forest that leaves `valid_trees = 0`, the
  // divisor is zero, and VI3 stays at its zero initialization.
  FeatureMatrix x = MAT(Feature, rows(4), 0.0F, 0.0F, 0.1F, 0.1F, 9.9F, 0.0F, 9.8F, 0.1F);
  OutcomeVector y = VEC(Outcome, 0, 0, 1, 1);

  ClassificationForest forest = single_split_forest({0, 1, 2, 3}); // all in-bag

  FeatureVector vi3 = vi_weighted_projections(forest, x, y.cast<Outcome>());

  ASSERT_NEAR(vi3(0), 0.0F, 1e-5F);
  ASSERT_NEAR(vi3(1), 0.0F, 1e-5F);
}

// ---------------------------------------------------------------------------
// ViPermuted — discriminating variable gets highest importance
// ---------------------------------------------------------------------------

TEST(ViPermuted, ClassificationDiscriminatingVariableTops) {
  auto [x, y] = perfect_separation_data();

  auto forest = Forest::train(TrainingSpec::builder(types::Mode::Classification).size(10).build(), x, y);


  FeatureVector vi1 = vi_permuted(forest, x, y.cast<Outcome>(), 0);

  ASSERT_GT(vi1(0), vi1(1)) << "Expected discriminating variable (col 0) to have higher VI1 than noise (col 1). "
                            << "VI1[0]=" << vi1(0) << ", VI1[1]=" << vi1(1);
}

// ---------------------------------------------------------------------------
// Regression VI — informative features should outrank noise
// ---------------------------------------------------------------------------

namespace {
  TrainingSpec make_reg_spec(int size, int n_vars_val, int seed) {
    return TrainingSpec::builder(types::Mode::Regression)
        .size(size)
        .seed(seed)
        .threads(1)
        .pp(pp::pda(Feature(0)))
        .vars(vars::uniform(n_vars_val))
        .grouping(grouping::by_cutpoint())
        .leaf(leaf::mean_response())
        .stop(stop::any({stop::min_size(5), stop::min_variance(0.01F)}))
        .build();
  }
}

TEST(ViPermuted, RegressionInformativeOutranksNoise) {
  // simulate generates y = sum(j+1 * x_j) for the first
  // n_informative columns plus noise. With n_informative=2 and p=5,
  // columns 0 and 1 drive y; columns 2..4 are pure noise.
  RNG rng(42);
  simulation::params::Regression params;
  params.n_informative = 2;
  params.y_sd          = 0.1F;
  params.sd            = 1.0F;
  auto data            = simulate(200, 5, rng, params);

  auto spec   = make_reg_spec(20, 3, 0);
  auto forest = Forest::train(spec, data.x, data.y);

  FeatureVector vi1 = vi_permuted(*forest, data.x, data.y, 0);

  // Both informative variables should outrank every noise variable.
  for (int j = 2; j < 5; ++j) {
    EXPECT_GT(vi1(0), vi1(j)) << "VI1[0]=" << vi1(0) << " not > VI1[" << j << "]=" << vi1(j);
    EXPECT_GT(vi1(1), vi1(j)) << "VI1[1]=" << vi1(1) << " not > VI1[" << j << "]=" << vi1(j);
  }
}

TEST(ViProjections, RegressionForestWithDegenerateSubBranchesStillScores) {
  // Small n, large p: every bootstrap sample of n=30, p=10 produces a
  // rank-deficient covariance somewhere down the tree, so most trees end
  // up with at least one aborted sub-branch and `bt->degenerate()` (which
  // propagates from any descendant leaf) is true on the whole forest.
  //
  // Regression: this used to make `vi_projections` return all zeros —
  // the whole-tree skip discarded every tree's still-valid upper splits.
  // Asserting that informative features come out non-zero locks in the
  // fix (the filter was dropped; degenerate leaves contribute nothing
  // and partial trees still contribute their valid branches).
  RNG rng(42);
  simulation::params::Regression params;
  params.n_informative = 2;
  params.y_sd          = 0.1F;
  params.sd            = 1.0F;
  auto data            = simulate(30, 10, rng, params);

  auto spec   = make_reg_spec(25, 3, 0);
  auto forest = Forest::train(spec, data.x, data.y);

  FeatureVector vi2 = vi_projections(*forest, static_cast<int>(data.x.cols()));

  // Informative columns 0 and 1 should both contribute non-zero VI even
  // though many of the 25 trees have aborted sub-branches deep down.
  EXPECT_GT(vi2(0), Feature(0)) << "VI2[0] should be > 0 even when most trees have degenerate sub-branches";
  EXPECT_GT(vi2(1), Feature(0)) << "VI2[1] should be > 0 even when most trees have degenerate sub-branches";
}

TEST(ViWeightedProjections, RegressionInformativeDominatesAsGroup) {
  RNG rng(42);
  simulation::params::Regression params;
  params.n_informative = 2;
  params.y_sd          = 0.1F;
  params.sd            = 1.0F;
  auto data            = simulate(200, 5, rng, params);

  auto spec   = make_reg_spec(20, 3, 0);
  auto forest = Forest::train(spec, data.x, data.y);

  FeatureVector scale = stats::sd(data.x);
  scale               = (scale.array() > Feature(0)).select(scale, Feature(1));

  FeatureVector vi3 = vi_weighted_projections(*forest, data.x, data.y, &scale);

  // VI3 is structural (projection coefficient × tree-quality weight); individual
  // noise features can occasionally outrank a single informative feature by chance.
  // The aggregate signal should still favour the informative set.
  Feature const total_informative = vi3(0) + vi3(1);
  Feature const total_noise       = vi3(2) + vi3(3) + vi3(4);

  EXPECT_GT(total_informative, total_noise)
      << "Sum(VI3[0..1])=" << total_informative << " not > Sum(VI3[2..4])=" << total_noise;
}

TEST(VariableImportance, BundleFillsAllFields) {
  RNG rng(42);
  auto data = simulate(100, 4, rng);

  auto spec   = make_reg_spec(5, 2, 0);
  auto forest = Forest::train(spec, data.x, data.y);

  VariableImportance vi = variable_importance(*forest, data.x, data.y, 0);

  int const p = static_cast<int>(data.x.cols());
  EXPECT_EQ(vi.permuted.size(), p);
  EXPECT_EQ(vi.projections.size(), p);
  EXPECT_EQ(vi.weighted_projections.size(), p);
  EXPECT_EQ(vi.scale.size(), p);
}
// ---------------------------------------------------------------------------
// OOB error
// ---------------------------------------------------------------------------

TEST(OOBError, PerfectSeparationGivesLowError) {
  auto [x, y] = perfect_separation_data();

  auto forest = ClassificationForest::train(TrainingSpec::builder(types::Mode::Classification).size(50).build(), x, y);

  auto const err = oob_error(*forest, x, y);

  ASSERT_TRUE(err.has_value());
  ASSERT_GE(*err, 0.0);
  ASSERT_LE(*err, 0.1) << "Expected near-zero OOB error for perfectly separable data";
}

TEST(OOBError, AllInBagReturnsNullopt) {
  FeatureMatrix x = MAT(Feature, rows(4), 0.0F, 0.0F, 0.1F, 0.1F, 9.9F, 0.0F, 9.8F, 0.1F);
  OutcomeVector y = VEC(Outcome, 0, 0, 1, 1);

  ClassificationForest forest = single_split_forest({0, 1, 2, 3});

  auto const err = oob_error(forest, x, y);

  ASSERT_FALSE(err.has_value()) << "No OOB observations: oob_error returns nullopt";
}

TEST(OOBError, HandBuiltTreeWithKnownOOB) {
  FeatureMatrix x = MAT(Feature, rows(6), 0.0F, 0.5F, 0.1F, 0.3F, 0.2F, 0.7F, 9.8F, 0.4F, 9.9F, 0.6F, 9.7F, 0.2F);
  OutcomeVector y = VEC(Outcome, 0, 0, 0, 1, 1, 1);

  ClassificationForest forest = single_split_forest({0, 1, 4, 5});

  auto const err = oob_error(forest, x, y);

  ASSERT_TRUE(err.has_value());
  ASSERT_DOUBLE_EQ(*err, 0.0);
}

TEST(OOBError, HandBuiltTreeWithOOBMisclassification) {
  FeatureMatrix x = MAT(Feature, rows(4), 0.0F, 0.0F, 0.1F, 0.1F, 9.9F, 0.0F, 9.8F, 0.1F);
  OutcomeVector y = VEC(Outcome, 0, 1, 1, 1);

  ClassificationForest forest = single_split_forest({0, 2});

  auto const err = oob_error(forest, x, y);

  ASSERT_TRUE(err.has_value());
  ASSERT_DOUBLE_EQ(*err, 0.5);
}

// ---------------------------------------------------------------------------
// oob_predict
// ---------------------------------------------------------------------------

TEST(OOBPredict, HandBuiltTreeWithKnownOOB) {
  FeatureMatrix x = MAT(Feature, rows(6), 0.0F, 0.5F, 0.1F, 0.3F, 0.2F, 0.7F, 9.8F, 0.4F, 9.9F, 0.6F, 9.7F, 0.2F);

  ClassificationForest forest = single_split_forest({0, 1, 4, 5});

  OutcomeVector preds = oob_predict(forest, x);

  ASSERT_EQ(preds.size(), 6);
  EXPECT_TRUE(std::isnan(preds(0))) << "Row 0 in bag";
  EXPECT_TRUE(std::isnan(preds(1))) << "Row 1 in bag";
  EXPECT_EQ(preds(2), 0) << "Row 2 OOB, x[0]=0.2 < 5";
  EXPECT_EQ(preds(3), 1) << "Row 3 OOB, x[0]=9.8 > 5";
  EXPECT_TRUE(std::isnan(preds(4))) << "Row 4 in bag";
  EXPECT_TRUE(std::isnan(preds(5))) << "Row 5 in bag";
}

TEST(OOBPredict, AllInBagReturnsSentinel) {
  // Both modes use `NaN` as the "no OOB tree" sentinel; classification
  // used to use `-1` but that complicated the R-side mapping (separate
  // filter path per mode). Unified on `NaN` so callers can use
  // `std::isnan` regardless of mode.
  FeatureMatrix x = MAT(Feature, rows(4), 0.0F, 0.0F, 0.1F, 0.1F, 9.9F, 0.0F, 9.8F, 0.1F);

  ClassificationForest forest = single_split_forest({0, 1, 2, 3});

  OutcomeVector const preds = oob_predict(forest, x);

  ASSERT_EQ(preds.size(), 4);
  EXPECT_TRUE(std::isnan(preds(0)));
  EXPECT_TRUE(std::isnan(preds(1)));
  EXPECT_TRUE(std::isnan(preds(2)));
  EXPECT_TRUE(std::isnan(preds(3)));
}

TEST(OOBPredict, HandBuiltTreeWithOOBMisclassification) {
  FeatureMatrix x = MAT(Feature, rows(4), 0.0F, 0.0F, 0.1F, 0.1F, 9.9F, 0.0F, 9.8F, 0.1F);

  ClassificationForest forest = single_split_forest({0, 2});

  OutcomeVector const preds = oob_predict(forest, x);

  ASSERT_EQ(preds.size(), 4);
  EXPECT_TRUE(std::isnan(preds(0))) << "Row 0 in bag";
  EXPECT_EQ(preds(1), 0) << "Row 1 OOB, x[0]=0.1 < 5 -> group 0";
  EXPECT_TRUE(std::isnan(preds(2))) << "Row 2 in bag";
  EXPECT_EQ(preds(3), 1) << "Row 3 OOB, x[0]=9.8 > 5 -> group 1";
}

TEST(OOBPredict, ConsistentWithOOBError) {
  auto [x, y] = perfect_separation_data();

  auto forest = ClassificationForest::train(TrainingSpec::builder(types::Mode::Classification).size(50).build(), x, y);

  OutcomeVector const preds = oob_predict(*forest, x);
  auto const err            = oob_error(*forest, x, y);

  int evaluated = 0;
  int correct   = 0;

  for (int i = 0; i < preds.size(); ++i) {
    // `NaN` is the "no OOB tree" sentinel; skip those rows. Any
    // non-NaN value is a valid (0-based) group id prediction.
    if (!std::isnan(preds(i))) {
      ++evaluated;

      if (preds(i) == y(i)) {
        ++correct;
      }
    }
  }

  if (evaluated == 0) {
    ASSERT_FALSE(err.has_value());
  } else {
    double const expected_err = 1.0 - static_cast<double>(correct) / static_cast<double>(evaluated);
    ASSERT_TRUE(err.has_value());
    ASSERT_DOUBLE_EQ(*err, expected_err) << "oob_error should match manual computation from oob_predict";
  }
}

// ---------------------------------------------------------------------------
// Regression OOB
// ---------------------------------------------------------------------------

TEST(RegressionForest, OOBPredict) {
  // Any forest produces an OOB prediction vector matching x's row count;
  // sentinel slots fill the rows that no tree leaves out of bag.
  FeatureMatrix x         = FeatureMatrix::Zero(10, 2);
  RegressionForest forest = leaf_regression_forest({{0, 1, 2, 3}});

  OutcomeVector oob_preds = oob_predict(forest, x);

  EXPECT_EQ(oob_preds.size(), x.rows());
}

TEST(RegressionForest, OOBError) {
  // Single-tree forest with bag={0..4}, leaf=5.0. OOB rows = {5..9}.
  // y(i) = i, so OOB squared errors are (5-5)², (5-6)², ..., (5-9)² = 0,1,4,9,16.
  // Expected MSE = 30/5 = 6.0.
  FeatureMatrix x = FeatureMatrix::Zero(10, 2);
  OutcomeVector y(10);
  for (int i = 0; i < 10; ++i) {
    y(i) = static_cast<Feature>(i);
  }

  RegressionForest forest = leaf_regression_forest({{0, 1, 2, 3, 4}}, /*leaf=*/5.0F);

  auto err = oob_error(forest, x, y);

  ASSERT_TRUE(err.has_value());
  EXPECT_NEAR(*err, 6.0, 1e-6);
}

TEST(RegressionForest, OOBWithManyTrees) {
  // 50 trees on 10 rows; each tree leaves a different row out of bag, so
  // every row has at least one OOB tree → no NaN sentinels in the output.
  int const n = 10;
  std::vector<std::vector<int>> bags;
  for (int t = 0; t < 50; ++t) {
    std::vector<int> bag;
    for (int i = 0; i < n; ++i) {
      if (i != t % n) {
        bag.push_back(i);
      }
    }
    bags.push_back(std::move(bag));
  }

  FeatureMatrix x         = FeatureMatrix::Zero(n, 2);
  RegressionForest forest = leaf_regression_forest(std::move(bags));

  OutcomeVector oob_preds = oob_predict(forest, x);

  int oob_count = 0;
  for (int i = 0; i < oob_preds.size(); ++i) {
    if (!std::isnan(static_cast<double>(oob_preds(i)))) {
      oob_count++;
    }
  }

  EXPECT_EQ(oob_count, n);
}

TEST(RegressionForest, OOBPredictSentinelIsNaNNotMinusOne) {
  // Two trees, both bagging every row → no OOB anywhere → every output is
  // the NaN sentinel. -1 used to be the sentinel but collides with valid
  // regression predictions of -1.0.
  int const n             = 10;
  std::vector<int> all_in = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  RegressionForest forest = leaf_regression_forest({all_in, all_in});
  FeatureMatrix x         = FeatureMatrix::Zero(n, 2);

  OutcomeVector oob_preds = oob_predict(forest, x);

  bool saw_sentinel = false;
  for (int i = 0; i < oob_preds.size(); ++i) {
    double v = static_cast<double>(oob_preds(i));
    if (std::isnan(v)) {
      saw_sentinel = true;
    } else {
      EXPECT_TRUE(std::isfinite(v));
    }
  }

  EXPECT_TRUE(saw_sentinel);
}

TEST(RegressionForest, OOBErrorRejectsWrongTruthSize) {
  FeatureMatrix x         = FeatureMatrix::Zero(10, 2);
  RegressionForest forest = leaf_regression_forest({{0, 1, 2, 3}});

  OutcomeVector wrong_size = OutcomeVector::Zero(5);

  EXPECT_THROW(oob_error(forest, x, wrong_size), std::exception);
}

TEST(RegressionForest, OOBErrorIncludesPredictionsThatEqualMinusOne) {
  // Single tree with leaf = -1.0F, bag={0..4}. OOB rows {5..9} all get
  // prediction -1.0. The old `-1` sentinel would drop these rows; the
  // NaN sentinel + std::optional contract both avoid that.
  FeatureMatrix x = FeatureMatrix::Zero(10, 2);
  OutcomeVector y = OutcomeVector::Zero(10); // truth = 0 → squared error = 1.0 per OOB row

  RegressionForest forest = leaf_regression_forest({{0, 1, 2, 3, 4}}, /*leaf=*/-1.0F);

  auto err = oob_error(forest, x, y);

  ASSERT_TRUE(err.has_value());
  EXPECT_NEAR(*err, 1.0, 1e-6);
}
