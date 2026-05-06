#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>

#include "models/Evaluation.hpp"
#include "models/Forest.hpp"
#include "models/TrainingSpec.hpp"
#include "stats/Metrics.hpp"
#include "stats/Stats.hpp"
#include "stats/Simulation.hpp"
#include "utils/Macros.hpp"

using namespace ppforest2;
using namespace ppforest2::types;
using namespace ppforest2::stats;
// ---------------------------------------------------------------------------
// RegressionForest tests
// ---------------------------------------------------------------------------

TEST(RegressionForest, TrainsSuccessfully) {
  RNG rng(0);
  auto data = simulate(40, 2, rng);
  auto spec = TrainingSpec::builder(types::Mode::Regression)
                  .pp(pp::pda(0.0F))
                  .vars(vars::all())
                  .grouping(grouping::by_cutpoint())
                  .leaf(leaf::mean_response())
                  .stop(stop::any({stop::min_size(5), stop::min_variance(0.001F)}))
                  .size(5)
                  .seed(0)
                  .threads(1)
                  .build();

  auto forest = Forest::train(spec, data.x, data.y);

  EXPECT_EQ(static_cast<int>(forest->trees.size()), 5);
}

TEST(RegressionForest, PredictsMatrix) {
  RNG rng(0);
  auto data = simulate(40, 2, rng);
  auto spec = TrainingSpec::builder(types::Mode::Regression)
                  .pp(pp::pda(0.0F))
                  .vars(vars::all())
                  .grouping(grouping::by_cutpoint())
                  .leaf(leaf::mean_response())
                  .stop(stop::any({stop::min_size(5), stop::min_variance(0.001F)}))
                  .size(5)
                  .seed(0)
                  .threads(1)
                  .build();

  auto forest = Forest::train(spec, data.x, data.y);

  OutcomeVector preds = forest->predict(data.x);

  EXPECT_EQ(preds.size(), data.x.rows());

  for (int i = 0; i < preds.size(); ++i) {
    EXPECT_TRUE(std::isfinite(preds(i)));
  }
}

TEST(RegressionForest, PredictsMean) {
  RNG rng(0);
  auto data = simulate(40, 2, rng);
  auto spec = TrainingSpec::builder(types::Mode::Regression)
                  .pp(pp::pda(0.0F))
                  .vars(vars::all())
                  .grouping(grouping::by_cutpoint())
                  .leaf(leaf::mean_response())
                  .stop(stop::any({stop::min_size(5), stop::min_variance(0.001F)}))
                  .size(5)
                  .seed(0)
                  .threads(1)
                  .build();

  auto forest = Forest::train(spec, data.x, data.y);

  // Forest regression prediction is the mean of tree predictions.
  OutcomeVector preds = forest->predict(data.x);

  RegressionMetrics metrics(preds, data.y);

  EXPECT_LT(metrics.mse, 1.0);
}


TEST(RegressionForest, Reproducible) {
  RNG rng(0);
  auto data  = simulate(30, 2, rng);
  auto spec1 = TrainingSpec::builder(types::Mode::Regression)
                   .pp(pp::pda(0.0F))
                   .vars(vars::all())
                   .grouping(grouping::by_cutpoint())
                   .leaf(leaf::mean_response())
                   .stop(stop::any({stop::min_size(5), stop::min_variance(0.001F)}))
                   .size(5)
                   .seed(0)
                   .threads(1)
                   .build();
  auto spec2 = TrainingSpec::builder(types::Mode::Regression)
                   .pp(pp::pda(0.0F))
                   .vars(vars::all())
                   .grouping(grouping::by_cutpoint())
                   .leaf(leaf::mean_response())
                   .stop(stop::any({stop::min_size(5), stop::min_variance(0.001F)}))
                   .size(5)
                   .seed(0)
                   .threads(1)
                   .build();

  auto f1 = Forest::train(spec1, data.x, data.y);
  auto f2 = Forest::train(spec2, data.x, data.y);

  OutcomeVector preds1 = f1->predict(data.x);
  OutcomeVector preds2 = f2->predict(data.x);

  for (int i = 0; i < preds1.size(); ++i) {
    EXPECT_FLOAT_EQ(preds1(i), preds2(i));
  }
}

TEST(RegressionForest, MultipleThreads) {
  RNG rng(0);
  auto data = simulate(100, 2, rng);

  auto spec = TrainingSpec::builder(types::Mode::Regression)
                  .pp(pp::pda(0.0F))
                  .vars(vars::uniform(2))
                  .grouping(grouping::by_cutpoint())
                  .leaf(leaf::mean_response())
                  .stop(stop::any({stop::min_size(5), stop::min_variance(0.001F)}))
                  .size(10)
                  .seed(0)
                  .threads(2)
                  .build();

  auto forest = Forest::train(spec, data.x, data.y);

  EXPECT_EQ(static_cast<int>(forest->trees.size()), 10);
}

TEST(RegressionForest, PredictsMeanOfTreePredictions) {
  RNG rng(0);
  auto data = simulate(40, 2, rng);
  auto spec = TrainingSpec::builder(types::Mode::Regression)
                  .pp(pp::pda(0.0F))
                  .vars(vars::all())
                  .grouping(grouping::by_cutpoint())
                  .leaf(leaf::mean_response())
                  .stop(stop::any({stop::min_size(5), stop::min_variance(0.001F)}))
                  .size(3)
                  .seed(0)
                  .threads(1)
                  .build();

  auto forest = Forest::train(spec, data.x, data.y);

  OutcomeVector forest_preds = forest->predict(data.x);

  for (int i = 0; i < data.x.rows(); ++i) {
    FeatureVector row = data.x.row(i);
    Feature sum       = 0;
    for (auto const& tree : forest->trees) {
      sum += static_cast<Feature>(tree->predict(row));
    }
    Feature expected_mean = sum / static_cast<Feature>(forest->trees.size());
    EXPECT_NEAR(forest_preds(i), expected_mean, 1e-4);
  }
}

TEST(RegressionForestSimulation, LowNoiseLowMSE) {
  // Analog of `ForestSimulation.PerfectSeparationLowOOBError`. The OOB
  // MSE should be a small fraction of the response variance — i.e.
  // the forest captures most of the signal, with residual variance
  // dominated by the (low) simulated noise plus tree-fit error.
  //
  // We compare to `Var(y)` rather than to an absolute MSE because the
  // response scale here is driven by deterministic coefficients
  // `(1, 2, 3, 4)` on the informative features, so absolute MSE is not
  // directly meaningful — the ratio is.
  RNG rng(0);
  simulation::params::Regression params;
  params.y_sd = 0.05F; // very low response noise

  auto data = simulate(200, 4, rng, params);

  auto forest = Forest::train(
      TrainingSpec::builder(types::Mode::Regression).size(20).threads(1).vars(vars::uniform(3)).build(), data.x, data.y
  );

  auto const err = oob_error(forest, data.x, data.y);

  ASSERT_TRUE(err.has_value());
  ASSERT_GE(*err, 0.0);

  double const y_var = static_cast<double>(var(data.y));
  ASSERT_GT(y_var, 0.0);
  // Forest should explain the bulk of the response variance. A
  // mean-only baseline would give err == Var(y); a perfect oracle
  // would give err == noise-variance. We require well under half of
  // baseline — generous enough to absorb seed-dependent fit error,
  // tight enough to catch a real regression.
  ASSERT_LT(*err / y_var, 0.5) << "Forest OOB MSE (" << *err << ") should be < 0.5 * Var(y) (" << y_var
                               << ") on low-noise simulated data";
}

TEST(RegressionForestSimulation, Deterministic) {
  // Same seed + same data should produce bit-identical predictions.
  // Mirrors `ForestSimulation.Deterministic` on the classification
  // side; load-bearing for reproducibility (cf. CLAUDE.md).
  RNG rng(0);
  auto data = simulate(100, 3, rng);

  auto f1 = Forest::train(
      TrainingSpec::builder(types::Mode::Regression).size(10).threads(1).vars(vars::uniform(3)).build(), data.x, data.y
  );
  auto f2 = Forest::train(
      TrainingSpec::builder(types::Mode::Regression).size(10).threads(1).vars(vars::uniform(3)).build(), data.x, data.y
  );

  OutcomeVector p1 = f1->predict(data.x);
  OutcomeVector p2 = f2->predict(data.x);

  ASSERT_EQ(p1.size(), p2.size());
  for (Eigen::Index i = 0; i < p1.size(); ++i) {
    EXPECT_FLOAT_EQ(p1(i), p2(i));
  }
}

TEST(RegressionForestSimulation, LargeDataset) {
  // Pure scaling sanity: large n × p must train + predict without
  // failure and produce a sensible fit. No tight error bound — just
  // making sure nothing crashes or returns garbage.
  RNG rng(0);
  simulation::params::Regression params;
  params.y_sd = 0.1F;

  auto data = simulate(2000, 10, rng, params);

  auto forest = Forest::train(
      TrainingSpec::builder(types::Mode::Regression).size(10).threads(1).vars(vars::uniform(5)).build(), data.x, data.y
  );

  OutcomeVector preds = forest->predict(data.x);
  ASSERT_EQ(preds.size(), data.x.rows());

  RegressionMetrics metrics(preds, data.y);
  EXPECT_TRUE(std::isfinite(metrics.mse));
}

// ---------------------------------------------------------------------------
// RegressionForestEdgeCase: structural edge cases parallel to the
// classification `ForestEdgeCase` block. Each test exercises a
// degenerate input shape and asserts the forest trains + predicts
// without crashing / returning non-finite values, rather than caring
// about prediction accuracy.
// ---------------------------------------------------------------------------

TEST(RegressionForestEdgeCase, ConstantFeatureColumn) {
  // First column is constant → the variable selector can pick it but
  // it carries no signal; the tree builder must still find a valid
  // split (or stop) on the remaining column.
  FeatureMatrix x = MAT(Feature, rows(6), 5, 1, 5, 2, 5, 3, 5, 7, 5, 8, 5, 9);
  OutcomeVector y = VEC(Outcome, 0.1F, 0.2F, 0.3F, 0.7F, 0.8F, 0.9F);

  auto forest =
      Forest::train(TrainingSpec::builder(types::Mode::Regression).size(5).threads(1).vars(vars::all()).build(), x, y);

  OutcomeVector const predictions = forest->predict(x);
  ASSERT_EQ(predictions.size(), y.size());
  for (Eigen::Index i = 0; i < predictions.size(); ++i) {
    EXPECT_TRUE(std::isfinite(predictions(i)));
  }
}

TEST(RegressionForestEdgeCase, ConstantResponse) {
  // `y` is constant → `min_variance` should fire at the root, making
  // every tree a single-leaf stump that returns the constant. Forest
  // mean prediction must equal that constant.
  FeatureMatrix x = MAT(Feature, rows(6), 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6);
  OutcomeVector y = VEC(Outcome, 7.5F, 7.5F, 7.5F, 7.5F, 7.5F, 7.5F);

  auto forest =
      Forest::train(TrainingSpec::builder(types::Mode::Regression).size(5).threads(1).vars(vars::all()).build(), x, y);

  OutcomeVector const predictions = forest->predict(x);
  ASSERT_EQ(predictions.size(), y.size());
  for (Eigen::Index i = 0; i < predictions.size(); ++i) {
    EXPECT_FLOAT_EQ(predictions(i), 7.5F);
  }
}

TEST(RegressionForestEdgeCase, MinimalDataset) {
  // n=2: ByCutpoint's `compute_init` returns `mid = 1`, splitting
  // into `[0,0]` / `[1,1]`. The forest must still train and produce
  // a finite prediction for each row.
  FeatureMatrix x = MAT(Feature, rows(2), 1, 9);
  OutcomeVector y = VEC(Outcome, 0.0F, 1.0F);

  auto forest =
      Forest::train(TrainingSpec::builder(types::Mode::Regression).size(5).threads(1).vars(vars::all()).build(), x, y);

  OutcomeVector const predictions = forest->predict(x);
  ASSERT_EQ(predictions.size(), y.size());
  for (Eigen::Index i = 0; i < predictions.size(); ++i) {
    EXPECT_TRUE(std::isfinite(predictions(i)));
  }
}
