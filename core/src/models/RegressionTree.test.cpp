#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>

#include "models/Model.hpp"
#include "models/RegressionTree.hpp"
#include "models/Tree.hpp"
#include "models/TrainingSpec.hpp"
#include "models/TreeLeaf.hpp"
#include "stats/Simulation.hpp"
#include "stats/Metrics.hpp"
#include "stats/Stats.hpp"
#include "test/TestSpec.hpp"
#include "utils/UserError.hpp"

using namespace ppforest2;
using namespace ppforest2::test;
using namespace ppforest2::types;
using namespace ppforest2::stats;

// ---------------------------------------------------------------------------
// RegressionTree tests
// ---------------------------------------------------------------------------

TEST(RegressionTree, TrainsSuccessfully) {
  RNG rng(0);
  auto data = simulate(30, 2, rng);
  auto spec = TrainingSpec::builder(types::Mode::Regression)
                  .pp(pp::pda(0.0F))
                  .grouping(grouping::by_cutpoint())
                  .leaf(leaf::mean_response())
                  .stop(stop::any({stop::min_size(5), stop::min_variance(0.001F)}))
                  .seed(0)
                  .build();

  auto tree = Tree::train(spec, data.x, data.y);

  EXPECT_NE(tree->root, nullptr);
  EXPECT_FALSE(tree->degenerate);
}

TEST(RegressionTree, PredictsSingleObservation) {
  RNG rng(0);
  auto data = simulate(30, 2, rng);
  auto spec = TrainingSpec::builder(types::Mode::Regression)
                  .pp(pp::pda(0.0F))
                  .grouping(grouping::by_cutpoint())
                  .leaf(leaf::mean_response())
                  .stop(stop::any({stop::min_size(5), stop::min_variance(0.001F)}))
                  .seed(0)
                  .build();

  auto tree = Tree::train(spec, data.x, data.y);

  Outcome pred = tree->predict(static_cast<FeatureVector>(data.x.row(0)));

  // Prediction should be a finite number (mean response of leaf).
  EXPECT_TRUE(std::isfinite(pred));
}

TEST(RegressionTree, PredictsMatrix) {
  RNG rng(0);
  auto data = simulate(30, 2, rng);
  auto spec = TrainingSpec::builder(types::Mode::Regression)
                  .pp(pp::pda(0.0F))
                  .grouping(grouping::by_cutpoint())
                  .leaf(leaf::mean_response())
                  .stop(stop::any({stop::min_size(5), stop::min_variance(0.001F)}))
                  .seed(0)
                  .build();

  auto tree = Tree::train(spec, data.x, data.y);

  OutcomeVector preds = tree->predict(data.x);

  EXPECT_EQ(preds.size(), data.x.rows());

  for (int i = 0; i < preds.size(); ++i) {
    EXPECT_TRUE(std::isfinite(preds(i)));
  }
}

TEST(RegressionTree, PredictionsAreReasonable) {
  RNG rng(0);
  auto data = simulate(50, 2, rng);
  auto spec = TrainingSpec::builder(types::Mode::Regression)
                  .pp(pp::pda(0.0F))
                  .grouping(grouping::by_cutpoint())
                  .leaf(leaf::mean_response())
                  .stop(stop::any({stop::min_size(5), stop::min_variance(0.001F)}))
                  .seed(0)
                  .build();

  auto tree = Tree::train(spec, data.x, data.y);

  OutcomeVector preds = tree->predict(data.x);

  // Training MSE should be relatively low for this simple dataset.
  RegressionMetrics metrics(preds, data.y);

  EXPECT_LT(metrics.mse, 1.0);
  EXPECT_GT(metrics.r_squared, -0.01);
}

TEST(RegressionTree, Reproducible) {
  RNG rng(0);
  auto data  = simulate(30, 2, rng);
  auto spec1 = TrainingSpec::builder(types::Mode::Regression)
                   .pp(pp::pda(0.0F))
                   .grouping(grouping::by_cutpoint())
                   .leaf(leaf::mean_response())
                   .stop(stop::any({stop::min_size(5), stop::min_variance(0.001F)}))
                   .seed(0)
                   .build();
  auto spec2 = TrainingSpec::builder(types::Mode::Regression)
                   .pp(pp::pda(0.0F))
                   .grouping(grouping::by_cutpoint())
                   .leaf(leaf::mean_response())
                   .stop(stop::any({stop::min_size(5), stop::min_variance(0.001F)}))
                   .seed(0)
                   .build();

  auto tree1 = Tree::train(spec1, data.x, data.y);
  auto tree2 = Tree::train(spec2, data.x, data.y);

  OutcomeVector preds1 = tree1->predict(data.x);
  OutcomeVector preds2 = tree2->predict(data.x);

  for (int i = 0; i < preds1.size(); ++i) {
    EXPECT_FLOAT_EQ(preds1(i), preds2(i));
  }
}

TEST(RegressionTree, ConstantResponse) {
  // All y values are the same; tree should immediately hit min_variance stop.
  int const n = 30;
  FeatureMatrix x(n, 2);
  OutcomeVector y(n);

  for (int i = 0; i < n; ++i) {
    x(i, 0) = static_cast<Feature>(i);
    x(i, 1) = static_cast<Feature>(n - i);
    y(i)    = 5.0F;
  }

  auto spec = TrainingSpec::builder(types::Mode::Regression)
                  .pp(pp::pda(0.0F))
                  .grouping(grouping::by_cutpoint())
                  .leaf(leaf::mean_response())
                  .stop(stop::any({stop::min_size(5), stop::min_variance(0.001F)}))
                  .seed(0)
                  .build();
  auto tree = Tree::train(spec, x, y);

  EXPECT_NE(tree->root, nullptr);

  OutcomeVector preds = tree->predict(x);
  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR(preds(i), 5.0, 1e-3);
  }
}

TEST(RegressionTree, LargeDataset) {
  int const n = 500;
  RNG rng(0);
  auto data = simulate(n, 2, rng);
  auto spec = TrainingSpec::builder(types::Mode::Regression)
                  .pp(pp::pda(0.0F))
                  .grouping(grouping::by_cutpoint())
                  .leaf(leaf::mean_response())
                  .stop(stop::any({stop::min_size(5), stop::min_variance(0.001F)}))
                  .seed(0)
                  .build();

  auto tree = Tree::train(spec, data.x, data.y);

  EXPECT_NE(tree->root, nullptr);

  OutcomeVector preds = tree->predict(data.x);
  EXPECT_EQ(preds.size(), n);

  RegressionMetrics metrics(preds, data.y);
  EXPECT_LT(metrics.mse, 1.0);
}

TEST(RegressionTree, MinSizeStop) {
  RNG rng(0);
  auto data = simulate(100, 2, rng);

  auto spec = TrainingSpec::builder(types::Mode::Regression)
                  .pp(pp::pda(0.0F))
                  .grouping(grouping::by_cutpoint())
                  .leaf(leaf::mean_response())
                  .stop(stop::min_size(50))
                  .seed(0)
                  .build();

  auto tree = Tree::train(spec, data.x, data.y);

  EXPECT_NE(tree->root, nullptr);
}

TEST(RegressionTree, MinVarianceStop) {
  RNG rng(0);
  auto data = simulate(100, 2, rng);

  auto spec = TrainingSpec::builder(types::Mode::Regression)
                  .pp(pp::pda(0.0F))
                  .grouping(grouping::by_cutpoint())
                  .leaf(leaf::mean_response())
                  .stop(stop::min_variance(10.0F))
                  .seed(0)
                  .build();

  auto tree = Tree::train(spec, data.x, data.y);

  EXPECT_NE(tree->root, nullptr);
}

TEST(RegressionTree, NegativeResponses) {
  int const n = 30;
  FeatureMatrix x(n, 2);
  OutcomeVector y(n);

  for (int i = 0; i < n; ++i) {
    x(i, 0) = static_cast<Feature>(i);
    x(i, 1) = static_cast<Feature>(n - i);
    y(i)    = static_cast<Feature>(i) - 15.0F;
  }

  auto spec = TrainingSpec::builder(types::Mode::Regression)
                  .pp(pp::pda(0.0F))
                  .grouping(grouping::by_cutpoint())
                  .leaf(leaf::mean_response())
                  .stop(stop::any({stop::min_size(5), stop::min_variance(0.001F)}))
                  .seed(0)
                  .build();
  auto tree = Tree::train(spec, x, y);

  EXPECT_NE(tree->root, nullptr);

  OutcomeVector preds = tree->predict(x);
  bool has_negative   = false;
  for (int i = 0; i < n; ++i) {
    if (preds(i) < 0) {
      has_negative = true;
      break;
    }
  }
  EXPECT_TRUE(has_negative);
}

TEST(RegressionTree, LargeResponseValues) {
  int const n = 30;
  FeatureMatrix x(n, 2);
  OutcomeVector y(n);

  for (int i = 0; i < n; ++i) {
    x(i, 0) = static_cast<Feature>(i);
    x(i, 1) = static_cast<Feature>(n - i);
    y(i)    = static_cast<Feature>(i) * 1000.0F;
  }

  auto spec = TrainingSpec::builder(types::Mode::Regression)
                  .pp(pp::pda(0.0F))
                  .grouping(grouping::by_cutpoint())
                  .leaf(leaf::mean_response())
                  .stop(stop::any({stop::min_size(5), stop::min_variance(0.001F)}))
                  .seed(0)
                  .build();
  auto tree = Tree::train(spec, x, y);

  OutcomeVector preds = tree->predict(x);
  for (int i = 0; i < n; ++i) {
    EXPECT_TRUE(std::isfinite(preds(i)));
  }
}

TEST(RegressionTree, NoProgressCutpointTerminates) {
  // If the cutpoint algorithm produces a split that leaves every row on one
  // side, the grouping strategy has no way to make progress. Prior to the
  // no-progress guard in build_root this would recurse without bound.
  //
  // Construct a weak stop rule (min_size only, no min_variance) and a small
  // near-constant-response dataset where the cutpoint and projector combination
  // could fail. Training must terminate rather than hang or overflow the stack.
  int const n = 20;
  FeatureMatrix x(n, 2);
  OutcomeVector y(n);

  for (int i = 0; i < n; ++i) {
    x(i, 0) = static_cast<Feature>(i);
    x(i, 1) = static_cast<Feature>(n - i);
    y(i)    = (i < n / 2) ? 0.0F : 1.0F; // bimodal; minimum viable regression target
  }

  auto spec = TrainingSpec::builder(types::Mode::Regression)
                  .pp(pp::pda(0.0F))
                  .grouping(grouping::by_cutpoint())
                  .leaf(leaf::mean_response())
                  .stop(stop::min_size(3)) // no min_variance
                  .seed(0)
                  .build();

  // The call must return; prior to the fix it could loop or stack overflow.
  auto tree = Tree::train(spec, x, y);
  EXPECT_NE(tree->root, nullptr);

  // Predictions should still be finite.
  OutcomeVector preds = tree->predict(x);
  for (int i = 0; i < n; ++i) {
    EXPECT_TRUE(std::isfinite(preds(i)));
  }
}

TEST(RegressionTree, BinaryPathTerminates) {
  int const n = 10;
  FeatureMatrix x(n, 2);
  OutcomeVector y(n);

  for (int i = 0; i < n; ++i) {
    x(i, 0) = static_cast<Feature>(i);
    x(i, 1) = static_cast<Feature>(n - i);
    y(i)    = static_cast<Feature>(i);
  }

  // min_size(n+1) causes immediate stop at root (MinSize stops when total < min_size).
  auto spec = TrainingSpec::builder(types::Mode::Regression)
                  .pp(pp::pda(0.0F))
                  .grouping(grouping::by_cutpoint())
                  .leaf(leaf::mean_response())
                  .stop(stop::min_size(n + 1))
                  .seed(0)
                  .build();

  auto tree = Tree::train(spec, x, y);

  EXPECT_NE(tree->root, nullptr);

  OutcomeVector preds  = tree->predict(x);
  double expected_mean = 0;
  for (int i = 0; i < n; ++i) {
    expected_mean += y(i);
  }
  expected_mean /= n;

  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR(preds(i), expected_mean, 1e-3);
  }
}

TEST(RegressionTree, PredictProportionsThrowsUserError) {
  // Vote proportions are classification-only — `predict_proportions` for a
  // regression model should raise `UserError` via the visitor's `visit(Tree)`
  // fallback.
  RegressionTree const tree(TreeLeaf::make(0.0F), test::regression_spec());

  FeatureMatrix x(2, 2);
  x << 1.0F, 2.0F, 3.0F, 4.0F;

  EXPECT_THROW(predict_proportions(tree, x), UserError);
}
