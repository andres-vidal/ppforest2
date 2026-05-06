#include <gtest/gtest.h>

#include "stats/Stats.hpp"
#include "utils/Types.hpp"

#include "utils/Macros.hpp"

using namespace ppforest2;
using namespace ppforest2::stats;
using namespace ppforest2::types;

TEST(Stats, Sort) {
  FeatureMatrix x = MAT(Feature, rows(3), 1.0, 3.0, 1.0, 2.0, 2.0, 3.0, 3.0, 1.0, 2.0);

  GroupIdVector y = VEC(GroupId, 1, 2, 1);

  sort(x, y);

  FeatureMatrix expected_x = MAT(Feature, rows(3), 1.0, 3.0, 1.0, 3.0, 1.0, 2.0, 2.0, 2.0, 3.0);

  GroupIdVector expected_y = VEC(GroupId, 1, 1, 2);

  ASSERT_EQ_DATA(expected_x, x);
  ASSERT_EQ_DATA(expected_y, y);
}

TEST(Stats, SortAlreadySorted) {
  FeatureMatrix x = MAT(Feature, rows(3), 1.0, 10.0, 2.0, 20.0, 3.0, 30.0);

  GroupIdVector y = VEC(GroupId, 0, 0, 1);

  sort(x, y);

  GroupIdVector expected_y = VEC(GroupId, 0, 0, 1);

  ASSERT_EQ(expected_y, y);
}

TEST(Stats, SortReverseSorted) {
  FeatureMatrix x = MAT(Feature, rows(4), 4.0, 3.0, 2.0, 1.0);

  GroupIdVector y = VEC(GroupId, 2, 1, 1, 0);

  sort(x, y);

  GroupIdVector expected_y = VEC(GroupId, 0, 1, 1, 2);

  ASSERT_EQ(expected_y, y);
  // Feature matrix should be reordered to match
  ASSERT_FLOAT_EQ(1.0, x(0, 0));
  ASSERT_FLOAT_EQ(4.0, x(3, 0));
}

TEST(Stats, SortLargerArray) {
  FeatureMatrix x = MAT(Feature, rows(6), 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);

  GroupIdVector y = VEC(GroupId, 2, 1, 0, 2, 1, 0);

  sort(x, y);

  for (int i = 1; i < y.size(); ++i) {
    ASSERT_LE(y[i - 1], y[i]);
  }
}

TEST(Stats, UniqueEmptyResult) {
  GroupIdVector column(0);
  std::set<int> actual = unique(column);
  std::set<int> expected;

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected, actual);
}

TEST(Stats, UniqueSingleValue) {
  GroupIdVector column   = VEC(GroupId, 1);
  std::set<int> actual   = unique(column);
  std::set<int> expected = {1};

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected, actual);
}

TEST(Stats, UniqueSingleValueRepeated) {
  GroupIdVector column = VEC(GroupId, 1, 1, 1);

  std::set<int> actual   = unique(column);
  std::set<int> expected = {1};

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected, actual);
}

TEST(Stats, UniqueMultipleValues) {
  GroupIdVector column   = VEC(GroupId, 1, 2, 3);
  std::set<int> actual   = unique(column);
  std::set<int> expected = {1, 2, 3};

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected, actual);
}

TEST(Stats, UniqueMultipleValuesRepeated) {
  GroupIdVector column   = VEC(GroupId, 1, 2, 1);
  std::set<int> actual   = unique(column);
  std::set<int> expected = {1, 2};

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected, actual);
}

TEST(Stats, SdVector) {
  FeatureVector v = VEC(Feature, 2.0F, 4.0F, 4.0F, 4.0F);
  // mean = 3.5, var = ((−1.5)²+(0.5)²+(0.5)²+(0.5)²) / 3 = 3/3 = 1, sd = 1
  ASSERT_NEAR(sd(v), 1.0, 1e-5);
}

TEST(Stats, SdVectorSingleElement) {
  FeatureVector v = VEC(Feature, 5.0F);
  ASSERT_DOUBLE_EQ(sd(v), 0.0);
}

TEST(Stats, SdMatrixSingleColumn) {
  FeatureMatrix m = MAT(Feature, rows(4), 2.0F, 4.0F, 4.0F, 4.0F);

  FeatureVector result = sd(m);

  ASSERT_EQ(result.size(), 1);
  ASSERT_NEAR(result(0), 1.0F, 1e-5F);
}

TEST(Stats, SdMatrixMultipleColumns) {
  // col 0: {1, 2, 3} → mean=2, var=1, sd=1
  // col 1: {4, 4, 4} → mean=4, var=0, sd=0
  // col 2: {0, 5, 10} → mean=5, var=25, sd=5
  FeatureMatrix m = MAT(Feature, rows(3), 1.0F, 4.0F, 0.0F, 2.0F, 4.0F, 5.0F, 3.0F, 4.0F, 10.0F);

  FeatureVector result = sd(m);

  ASSERT_EQ(result.size(), 3);
  ASSERT_NEAR(result(0), 1.0F, 1e-5F);
  ASSERT_NEAR(result(1), 0.0F, 1e-5F);
  ASSERT_NEAR(result(2), 5.0F, 1e-5F);
}

TEST(Stats, SdMatrixMatchesVectorSd) {
  FeatureMatrix m = MAT(Feature, rows(5), 1.0F, 10.0F, 3.0F, 20.0F, 5.0F, 30.0F, 7.0F, 40.0F, 9.0F, 50.0F);

  FeatureVector result = sd(m);

  ASSERT_NEAR(result(0), static_cast<float>(sd(static_cast<FeatureVector>(m.col(0)))), 1e-5F);
  ASSERT_NEAR(result(1), static_cast<float>(sd(static_cast<FeatureVector>(m.col(1)))), 1e-5F);
}

// ---------------------------------------------------------------------------
// var (vector and matrix overloads)
// ---------------------------------------------------------------------------

TEST(Stats, VarVector) {
  // mean = 3.5, var = ((−1.5)² + (0.5)² + (0.5)² + (0.5)²) / 3 = 3/3 = 1
  FeatureVector v = VEC(Feature, 2.0F, 4.0F, 4.0F, 4.0F);

  ASSERT_NEAR(var(v), 1.0, 1e-5);
}

TEST(Stats, VarVectorSingleObservation) {
  // Single observation: variance undefined, returns 0 by convention.
  FeatureVector v = VEC(Feature, 3.0F);

  ASSERT_DOUBLE_EQ(var(v), 0.0);
}

TEST(Stats, VarVectorZeroVariance) {
  FeatureVector v = VEC(Feature, 7.0F, 7.0F, 7.0F, 7.0F);

  ASSERT_DOUBLE_EQ(var(v), 0.0);
}

TEST(Stats, VarVectorEmptyThrows) {
  FeatureVector v(0);

  ASSERT_THROW(var(v), std::invalid_argument);
}

TEST(Stats, VarMatrixPerColumn) {
  // col 0: {1, 2, 3} → var = 1
  // col 1: {4, 4, 4} → var = 0
  // col 2: {0, 5, 10} → var = 25
  FeatureMatrix m = MAT(Feature, rows(3), 1.0F, 4.0F, 0.0F, 2.0F, 4.0F, 5.0F, 3.0F, 4.0F, 10.0F);

  FeatureVector result = var(m);

  ASSERT_EQ(result.size(), 3);
  ASSERT_NEAR(result(0), 1.0F, 1e-5F);
  ASSERT_NEAR(result(1), 0.0F, 1e-5F);
  ASSERT_NEAR(result(2), 25.0F, 1e-5F);
}

TEST(Stats, SdEqualsSqrtOfVar) {
  // Pin the identity so `sd` delegating to `var` stays honest.
  FeatureVector v = VEC(Feature, 1.0F, 2.0F, 4.0F, 8.0F, 16.0F);

  ASSERT_NEAR(sd(v), std::sqrt(var(v)), 1e-10);
}

TEST(Stats, SdMatrixEqualsSqrtOfVarMatrix) {
  FeatureMatrix m = MAT(Feature, rows(3), 1.0F, 4.0F, 0.0F, 2.0F, 4.0F, 5.0F, 3.0F, 4.0F, 10.0F);

  FeatureVector sd_result  = sd(m);
  FeatureVector var_result = var(m);

  ASSERT_NEAR(sd_result(0), std::sqrt(var_result(0)), 1e-5F);
  ASSERT_NEAR(sd_result(1), std::sqrt(var_result(1)), 1e-5F);
  ASSERT_NEAR(sd_result(2), std::sqrt(var_result(2)), 1e-5F);
}

// ---------------------------------------------------------------------------
// majority_vote
// ---------------------------------------------------------------------------

TEST(Stats, MajorityVoteSingleValue) {
  ASSERT_FLOAT_EQ(majority_vote({Outcome(7)}), Outcome(7));
}

TEST(Stats, MajorityVoteUnanimous) {
  ASSERT_FLOAT_EQ(majority_vote({Outcome(2), Outcome(2), Outcome(2)}), Outcome(2));
}

TEST(Stats, MajorityVoteClearWinner) {
  // 0:3, 1:1, 2:1 → 0 wins.
  ASSERT_FLOAT_EQ(majority_vote({Outcome(0), Outcome(1), Outcome(0), Outcome(2), Outcome(0)}), Outcome(0));
}

TEST(Stats, MajorityVoteTiesGoToSmallestGroupId) {
  // 1:2, 2:2, 3:2 — tie. `std::map` iterates ascending and `>` keeps the
  // earliest winner, so the smallest GroupId (1) wins.
  ASSERT_FLOAT_EQ(majority_vote({Outcome(3), Outcome(2), Outcome(1), Outcome(3), Outcome(2), Outcome(1)}), Outcome(1));
}

TEST(Stats, MajorityVoteTieBreakHonorsAscendingOrder) {
  // 0:1, 5:1 — tie. Should pick 0, not 5.
  ASSERT_FLOAT_EQ(majority_vote({Outcome(5), Outcome(0)}), Outcome(0));
}

TEST(Stats, MajorityVoteEmptyThrows) {
  ASSERT_THROW(majority_vote({}), std::runtime_error);
}

// ---------------------------------------------------------------------------
// mean
// ---------------------------------------------------------------------------

TEST(Stats, MeanSingleValue) {
  ASSERT_FLOAT_EQ(mean({Outcome(3.5F)}), Outcome(3.5F));
}

TEST(Stats, MeanSimpleAverage) {
  // (1 + 2 + 3 + 4) / 4 = 2.5
  ASSERT_FLOAT_EQ(mean({Outcome(1), Outcome(2), Outcome(3), Outcome(4)}), Outcome(2.5F));
}

TEST(Stats, MeanWithNegativeValues) {
  // (-1 + 0 + 1) / 3 = 0
  ASSERT_FLOAT_EQ(mean({Outcome(-1), Outcome(0), Outcome(1)}), Outcome(0));
}

TEST(Stats, MeanLargeSumNoFloatDrift) {
  // Sum is computed in `double` to avoid loss-of-precision when many
  // float outcomes are added. Verify the canonical behaviour with a
  // case that would drift in single-precision accumulation.
  std::vector<Outcome> preds(10000, Outcome(0.1F));
  ASSERT_NEAR(mean(preds), Outcome(0.1F), 1e-5F);
}

TEST(Stats, MeanEmptyThrows) {
  ASSERT_THROW(mean({}), std::runtime_error);
}

// ---------------------------------------------------------------------------
// group_indices
// ---------------------------------------------------------------------------

TEST(Stats, GroupIndicesEmpty) {
  std::map<GroupId, int> const indices = group_indices({});
  ASSERT_TRUE(indices.empty());
}

TEST(Stats, GroupIndicesSingleGroup) {
  std::map<GroupId, int> const indices = group_indices({7});
  ASSERT_EQ(indices.size(), 1U);
  ASSERT_EQ(indices.at(7), 0);
}

TEST(Stats, GroupIndicesAssignsAscendingColumns) {
  // `std::set` iterates in ascending key order, so columns are sorted by
  // GroupId regardless of the literal initialiser order.
  std::map<GroupId, int> const indices = group_indices({4, 1, 3, 2});

  ASSERT_EQ(indices.size(), 4U);
  ASSERT_EQ(indices.at(1), 0);
  ASSERT_EQ(indices.at(2), 1);
  ASSERT_EQ(indices.at(3), 2);
  ASSERT_EQ(indices.at(4), 3);
}

TEST(Stats, GroupIndicesMissingKeyThrowsAt) {
  // Documents the expected lookup behaviour callers rely on
  // (`indices.at(pred)` for one-hot column lookup).
  std::map<GroupId, int> const indices = group_indices({0, 2});
  ASSERT_THROW(static_cast<void>(indices.at(1)), std::out_of_range);
}
