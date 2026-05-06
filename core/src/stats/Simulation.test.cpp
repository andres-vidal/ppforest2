#include <gtest/gtest.h>

#include "stats/Simulation.hpp"
#include "stats/GroupPartition.hpp"

using namespace ppforest2;
using namespace ppforest2::types;
using namespace ppforest2::stats;

TEST(Simulate, ClassificationCorrectDimensions) {
  RNG rng(0);
  auto data = simulate(100, 4, 3, rng);

  ASSERT_EQ(data.x.rows(), 100);
  ASSERT_EQ(data.x.cols(), 4);
  ASSERT_EQ(data.y.size(), 100);
}

TEST(Simulate, ClassificationCorrectNumberOfClasses) {
  RNG rng(0);
  auto data = simulate(90, 4, 3, rng);

  ASSERT_EQ(data.groups.size(), 3);
  ASSERT_TRUE(data.groups.count(0));
  ASSERT_TRUE(data.groups.count(1));
  ASSERT_TRUE(data.groups.count(2));
}

TEST(Simulate, ClassificationBalancedClasses) {
  RNG rng(0);
  auto data = simulate(90, 4, 3, rng);

  GroupPartition spec(data.y);
  ASSERT_EQ(spec.group_size(0), 30);
  ASSERT_EQ(spec.group_size(1), 30);
  ASSERT_EQ(spec.group_size(2), 30);
}

TEST(Simulate, ClassificationSortedByLabel) {
  RNG rng(0);
  auto data = simulate(90, 4, 3, rng);

  for (int i = 1; i < data.y.size(); ++i) {
    ASSERT_LE(data.y[i - 1], data.y[i]);
  }
}

TEST(Simulate, ClassificationTwoClasses) {
  RNG rng(0);
  auto data = simulate(50, 2, 2, rng);

  ASSERT_EQ(data.x.rows(), 50);
  ASSERT_EQ(data.groups.size(), 2);
}

TEST(Simulate, ClassificationManyClasses) {
  RNG rng(0);
  auto data = simulate(100, 4, 10, rng);

  ASSERT_EQ(data.groups.size(), 10);
  ASSERT_EQ(data.x.rows(), 100);
}

TEST(Simulate, ClassificationSingleFeature) {
  RNG rng(0);
  auto data = simulate(50, 1, 2, rng);

  ASSERT_EQ(data.x.cols(), 1);
}

TEST(Simulate, ClassificationCustomParams) {
  RNG rng(0);
  simulation::params::Classification params;
  params.mean            = 0.0F;
  params.mean_separation = 100.0F;
  params.sd              = 1.0F;

  auto data = simulate(60, 2, 3, rng, params);

  ASSERT_EQ(data.x.rows(), 60);
  ASSERT_EQ(data.groups.size(), 3);
}

TEST(Simulate, ClassificationDeterministic) {
  RNG rng1(0);
  auto data1 = simulate(50, 4, 3, rng1);

  RNG rng2(0);
  auto data2 = simulate(50, 4, 3, rng2);

  ASSERT_EQ(data1.x, data2.x);
  ASSERT_EQ(data1.y, data2.y);
}

TEST(Simulate, ClassificationDifferentSeedsDifferentData) {
  RNG rng1(0);
  auto data1 = simulate(50, 4, 3, rng1);

  RNG rng2(99);
  auto data2 = simulate(50, 4, 3, rng2);

  ASSERT_NE(data1.x, data2.x);
}


TEST(Split, ClassificationPreservesClassProportions) {
  RNG rng(0);
  auto data = simulate(90, 4, 3, rng);

  RNG split_rng(99);
  auto s = split(data, 0.8f, split_rng);

  ASSERT_EQ(s.tr.size() + s.te.size(), 90);

  // With 30 per group and 0.8 ratio, expect ~24 train per group
  // Total train should be ~72
  ASSERT_GE(s.tr.size(), 60);
  ASSERT_LE(s.tr.size(), 78);
}

TEST(Split, ClassificationIndicesAreValid) {
  RNG rng(0);
  auto data = simulate(60, 4, 3, rng);

  RNG split_rng(0);
  auto s = split(data, 0.7F, split_rng);

  for (int idx : s.tr) {
    ASSERT_GE(idx, 0);
    ASSERT_LT(idx, 60);
  }

  for (int idx : s.te) {
    ASSERT_GE(idx, 0);
    ASSERT_LT(idx, 60);
  }
}

TEST(Split, ClassificationNoOverlapBetweenTrainAndTest) {
  RNG rng(0);
  auto data = simulate(60, 4, 3, rng);

  RNG split_rng(0);
  auto s = split(data, 0.7F, split_rng);

  std::set<int> train_set(s.tr.begin(), s.tr.end());
  for (int idx : s.te) {
    ASSERT_EQ(train_set.count(idx), 0);
  }
}

TEST(Split, ClassificationDeterministic) {
  RNG rng1(0);
  auto data = simulate(60, 4, 3, rng1);

  RNG split_rng1(0);
  auto s1 = split(data, 0.7F, split_rng1);

  RNG split_rng2(0);
  auto s2 = split(data, 0.7F, split_rng2);

  ASSERT_EQ(s1.tr, s2.tr);
  ASSERT_EQ(s1.te, s2.te);
}

TEST(Split, RegressionProducesNonEmptySides) {
  RNG rng(0);
  auto data = simulate(60, 4, rng);

  RNG split_rng(0);
  auto s = split(data, 0.7F, split_rng);

  ASSERT_EQ(static_cast<int>(s.tr.size() + s.te.size()), 60);
  ASSERT_GT(s.tr.size(), 0U) << "regression split must produce non-empty train";
  ASSERT_GT(s.te.size(), 0U) << "regression split must produce non-empty test";
}

TEST(Split, RegressionIndicesAreSortedPerSide) {
  RNG rng(0);
  auto data = simulate(40, 3, rng);

  RNG split_rng(0);
  auto s = split(data, 0.6F, split_rng);

  for (std::size_t i = 1; i < s.tr.size(); ++i) {
    ASSERT_LT(s.tr[i - 1], s.tr[i]);
  }
  for (std::size_t i = 1; i < s.te.size(); ++i) {
    ASSERT_LT(s.te[i - 1], s.te[i]);
  }
}

TEST(Split, RegressionNoOverlap) {
  RNG rng(0);
  auto data = simulate(60, 4, rng);

  RNG split_rng(0);
  auto s = split(data, 0.7F, split_rng);

  std::set<int> train_set(s.tr.begin(), s.tr.end());
  for (int idx : s.te) {
    ASSERT_EQ(train_set.count(idx), 0) << "row " << idx << " in both tr and te";
  }
}

TEST(Split, RegressionDeterministic) {
  RNG rng(0);
  auto data = simulate(50, 3, rng);

  RNG split_rng1(0);
  auto s1 = split(data, 0.7F, split_rng1);
  RNG split_rng2(0);
  auto s2 = split(data, 0.7F, split_rng2);

  ASSERT_EQ(s1.tr, s2.tr);
  ASSERT_EQ(s1.te, s2.te);
}


TEST(Simulate, RegressionCorrectDimensions) {
  RNG rng(0);
  auto data = simulate(100, 4, rng);

  ASSERT_EQ(data.x.rows(), 100);
  ASSERT_EQ(data.x.cols(), 4);
  ASSERT_EQ(data.y.size(), 100);
  ASSERT_TRUE(data.groups.empty());
}

TEST(Simulate, RegressionSortedByY) {
  RNG rng(0);
  auto data = simulate(50, 3, rng);

  for (int i = 1; i < 50; ++i) {
    ASSERT_LE(data.y(i - 1), data.y(i));
  }
}

TEST(Simulate, RegressionDeterministic) {
  RNG rng1(0);
  RNG rng2(0);

  auto d1 = simulate(50, 3, rng1);
  auto d2 = simulate(50, 3, rng2);

  ASSERT_EQ(d1.x.rows(), d2.x.rows());
  for (int i = 0; i < d1.x.rows(); ++i) {
    ASSERT_FLOAT_EQ(d1.y(i), d2.y(i));
    for (int j = 0; j < d1.x.cols(); ++j) {
      ASSERT_FLOAT_EQ(d1.x(i, j), d2.x(i, j));
    }
  }
}

TEST(Simulate, RegressionRespondsToInformativeFeatures) {
  // With only 2 informative features, the first 2 columns should correlate
  // strongly with y, while the rest should not.
  simulation::params::Regression params;
  params.n_informative = 2;
  params.y_sd          = 0.1F;

  RNG rng(0);
  auto data = simulate(500, 6, rng, params);

  // Correlation of column 0 with y should be higher than column 5.
  auto corr = [&](int col) {
    double mx = 0;
    double my = 0;
    for (int i = 0; i < 500; ++i) {
      mx += data.x(i, col);
      my += data.y(i);
    }
    mx /= 500;
    my /= 500;

    double num = 0;
    double dx  = 0;
    double dy  = 0;
    for (int i = 0; i < 500; ++i) {
      double a = data.x(i, col) - mx;
      double b = data.y(i) - my;
      num += a * b;
      dx += a * a;
      dy += b * b;
    }

    return std::abs(num) / std::sqrt(dx * dy);
  };

  double corr_informative = corr(0);
  double corr_noise       = corr(5);

  EXPECT_GT(corr_informative, corr_noise);
  EXPECT_GT(corr_informative, 0.2);
}
