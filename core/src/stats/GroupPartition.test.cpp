#include <gtest/gtest.h>

#include "stats/GroupPartition.hpp"
#include "utils/Types.hpp"

#include "utils/Macros.hpp"

using namespace ppforest2;
using namespace ppforest2::stats;
using namespace ppforest2::types;

TEST(GroupPartition, GroupSize) {
  FeatureMatrix x = MAT(Feature, rows(6), 2, 2, 2, 4, 4, 4, 1, 1, 1, 6, 6, 6, 3, 3, 3, 5, 5, 5);

  GroupPartition const y(VEC(GroupId, 1, 1, 2, 2, 3, 3));

  EXPECT_EQ(2, y.group_size(1));
  EXPECT_EQ(2, y.group_size(2));
  EXPECT_EQ(2, y.group_size(3));
}

TEST(GroupPartition, GroupStart) {
  FeatureMatrix x = MAT(Feature, rows(6), 2, 2, 2, 4, 4, 4, 1, 1, 1, 6, 6, 6, 3, 3, 3, 5, 5, 5);

  GroupPartition const y(VEC(GroupId, 1, 1, 2, 2, 3, 3));

  EXPECT_EQ(0, y.group_start(1));
  EXPECT_EQ(2, y.group_start(2));
  EXPECT_EQ(4, y.group_start(3));
}

TEST(GroupPartition, GroupEnd) {
  FeatureMatrix x = MAT(Feature, rows(6), 2, 2, 2, 4, 4, 4, 1, 1, 1, 6, 6, 6, 3, 3, 3, 5, 5, 5);

  GroupPartition const y(VEC(GroupId, 1, 1, 2, 2, 3, 3));

  EXPECT_EQ(1, y.group_end(1));
  EXPECT_EQ(3, y.group_end(2));
  EXPECT_EQ(5, y.group_end(3));
}

TEST(GroupPartition, Group) {
  FeatureMatrix x = MAT(Feature, rows(6), 2, 2, 2, 4, 4, 4, 1, 1, 1, 6, 6, 6, 3, 3, 3, 5, 5, 5);

  GroupPartition const y(VEC(GroupId, 1, 1, 2, 2, 3, 3));

  EXPECT_EQ_DATA(y.group(x, 1), MAT(Feature, rows(2), 2, 2, 2, 4, 4, 4));
  EXPECT_EQ_DATA(y.group(x, 2), MAT(Feature, rows(2), 1, 1, 1, 6, 6, 6));
  EXPECT_EQ_DATA(y.group(x, 3), MAT(Feature, rows(2), 3, 3, 3, 5, 5, 5));
}

TEST(GroupPartition, ErrorGroupsNotContiguous) {
  FeatureMatrix x       = MAT(Feature, rows(6), 2, 2, 2, 4, 4, 4, 1, 1, 1, 6, 6, 6, 3, 3, 3, 5, 5, 5);
  GroupIdVector const y = VEC(GroupId, 1, 2, 3, 1, 2, 3);

  ASSERT_THROW((GroupPartition(y)), std::invalid_argument);
}

TEST(GroupPartition, Subset) {
  FeatureMatrix x = MAT(Feature, rows(6), 1, 2, 2, 1, 4, 4, 2, 1, 1, 2, 6, 6, 3, 3, 3, 3, 5, 5);

  GroupPartition const y      = GroupPartition(VEC(GroupId, 1, 1, 2, 2, 3, 3));
  GroupPartition const subset = y.subset({1, 3});

  EXPECT_EQ(subset.groups, (std::set<int>{1, 3}));

  EXPECT_EQ(subset.group_size(1), 2);
  EXPECT_EQ(subset.group_start(1), 0);
  EXPECT_EQ(subset.group_end(1), 1);

  EXPECT_EQ(subset.group_size(3), 2);
  EXPECT_EQ(subset.group_start(3), 4);
  EXPECT_EQ(subset.group_end(3), 5);

  EXPECT_EQ_DATA(subset.group(x, 1), MAT(Feature, rows(2), 1, 2, 2, 1, 4, 4));
  EXPECT_EQ_DATA(subset.group(x, 3), MAT(Feature, rows(2), 3, 3, 3, 3, 5, 5));

  EXPECT_EQ_DATA(subset.data(x), MAT(Feature, rows(4), 1, 2, 2, 1, 4, 4, 3, 3, 3, 3, 5, 5));
  EXPECT_EQ(subset.mean(x), VEC(Feature, 2.0, 3.5, 3.5));
}

TEST(GroupPartition, Remap) {
  FeatureMatrix x = MAT(Feature, rows(6), 1, 1, 1, 1, 2, 2, 2, 1, 1, 2, 2, 2, 3, 1, 1, 3, 2, 2);

  GroupPartition const y(VEC(GroupId, 1, 1, 2, 2, 3, 3));
  GroupPartition const remapped = y.remap({{1, 0}, {2, 1}, {3, 0}});

  EXPECT_EQ(std::set<int>({0, 1}), remapped.groups);
  EXPECT_EQ_DATA(remapped.data(x), x);
}

TEST(GroupPartition, BetweenGroupsSumOfSquaresSingleGroup) {
  FeatureMatrix x = MAT(Feature, rows(3), 1.0, 2.0, 6.0, 2.0, 3.0, 7.0, 3.0, 4.0, 8.0);

  GroupPartition const y(VEC(GroupId, 0, 0, 0));

  EXPECT_EQ_DATA(y.bgss(x), MAT(Feature, rows(3), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
}

TEST(GroupPartition, BetweenGroupsSumOfSquaresTwoEqualGroups) {
  FeatureMatrix x =
      MAT(Feature, rows(6), 1.0, 2.0, 6.0, 2.0, 3.0, 7.0, 3.0, 4.0, 8.0, 1.0, 2.0, 6.0, 2.0, 3.0, 7.0, 3.0, 4.0, 8.0);

  GroupPartition const y(VEC(GroupId, 0, 0, 0, 1, 1, 1));

  EXPECT_EQ_DATA(y.bgss(x), MAT(Feature, rows(3), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
}

TEST(GroupPartition, BetweenGroupsSumOfSquaresMultipleGroupsUnivariate) {
  FeatureMatrix x = MAT(Feature, rows(8), 23.0, 25.0, 18.0, 29.0, 19.0, 21.0, 35.0, 17.0);

  GroupPartition const y(VEC(GroupId, 0, 0, 0, 1, 1, 1, 2, 2));

  EXPECT_EQ_DATA(y.bgss(x), MAT(Feature, rows(1), 19.875));
}

TEST(GroupPartition, BetweenGroupsSumOfSquaresMultipleGroupsUnivariateNonSequentialGroups) {
  FeatureMatrix x = MAT(Feature, rows(8), 23.0, 25.0, 18.0, 29.0, 19.0, 21.0, 35.0, 17.0);

  GroupPartition const y(VEC(GroupId, 1, 1, 1, 7, 7, 7, 3, 3));

  EXPECT_EQ_DATA(y.bgss(x), MAT(Feature, rows(1), 19.875));
}

TEST(GroupPartition, BetweenGroupsSumOfSquaresMultipleGroupsMultivariate) {
  FeatureMatrix x =
      MAT(Feature,
          rows(8),
          23.0,
          1.0,
          1.0,
          25.0,
          1.0,
          1.0,
          18.0,
          1.0,
          1.0,
          29.0,
          1.0,
          1.0,
          19.0,
          1.0,
          1.0,
          21.0,
          1.0,
          1.0,
          35.0,
          1.0,
          1.0,
          17.0,
          1.0,
          1.0);

  GroupPartition const y(VEC(GroupId, 0, 0, 0, 1, 1, 1, 2, 2));

  EXPECT_EQ_DATA(y.bgss(x), MAT(Feature, rows(3), 19.875, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
}

TEST(GroupPartition, WithinGroupsSumOfSquaresSingleGroupNoVariance) {
  FeatureMatrix x = MAT(Feature, rows(3), 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);

  GroupPartition y(VEC(GroupId, 0, 0, 0));

  EXPECT_EQ_DATA(y.wgss(x), MAT(Feature, rows(3), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
}

TEST(GroupPartition, WithinGroupsSumOfSquaresSingleGroupWithVariance) {
  FeatureMatrix x = MAT(Feature, rows(3), 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0);

  GroupPartition y(VEC(GroupId, 0, 0, 0));

  EXPECT_EQ_DATA(y.wgss(x), MAT(Feature, rows(3), 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0));
}

TEST(GroupPartition, WithinGroupsSumOfSquaresTwoEqualGroups) {
  FeatureMatrix x =
      MAT(Feature, rows(6), 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0);

  GroupPartition y(VEC(GroupId, 0, 0, 0, 1, 1, 1));


  EXPECT_EQ_DATA(y.wgss(x), MAT(Feature, rows(3), 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0));
}

TEST(GroupPartition, WithinGroupsSumOfSquaresTwoGroupsSameVariance) {
  FeatureMatrix x =
      MAT(Feature, rows(6), 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0);

  GroupPartition const y(VEC(GroupId, 0, 0, 0, 1, 1, 1));

  EXPECT_EQ_DATA(y.wgss(x), MAT(Feature, rows(3), 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0));
}

TEST(GroupPartition, WithinGroupsSumOfSquaresTwoGroupsDifferentVariance) {
  FeatureMatrix x =
      MAT(Feature, rows(6), 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0);

  GroupPartition const y(VEC(GroupId, 0, 0, 0, 1, 1, 1));

  EXPECT_EQ_DATA(y.wgss(x), MAT(Feature, rows(3), 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0, 16.0));
}

TEST(GroupPartition, WithinGroupsSumOfSquaresMultipleGroupsMultivariate1) {
  FeatureMatrix x =
      MAT(Feature,
          rows(8),
          1.0,
          2.0,
          3.0,
          4.0,
          5.0,
          6.0,
          7.0,
          8.0,
          9.0,
          3.0,
          2.0,
          1.0,
          4.0,
          3.0,
          2.0,
          5.0,
          4.0,
          3.0,
          9.0,
          8.0,
          7.0,
          6.0,
          5.0,
          4.0);

  GroupPartition y(VEC(GroupId, 0, 0, 0, 1, 1, 1, 2, 2));

  EXPECT_EQ_DATA(y.wgss(x), MAT(Feature, rows(3), 24.5, 24.5, 24.5, 24.5, 24.5, 24.5, 24.5, 24.5, 24.5));
}

TEST(GroupPartition, WithinGroupsSumOfSquaresMultipleGroupsMultivariate2) {
  FeatureMatrix x =
      MAT(Feature,
          rows(8),
          1.0,
          2.0,
          3.0,
          0.0,
          4.0,
          5.0,
          6.0,
          0.0,
          7.0,
          8.0,
          9.0,
          0.0,
          3.0,
          2.0,
          1.0,
          0.0,
          4.0,
          3.0,
          2.0,
          0.0,
          5.0,
          4.0,
          3.0,
          0.0,
          9.0,
          8.0,
          7.0,
          0.0,
          6.0,
          5.0,
          4.0,
          0.0);

  GroupPartition const y(VEC(GroupId, 0, 0, 0, 1, 1, 1, 2, 2));

  EXPECT_EQ_DATA(
      y.wgss(x),
      MAT(Feature, rows(4), 24.5, 24.5, 24.5, 0.0, 24.5, 24.5, 24.5, 0.0, 24.5, 24.5, 24.5, 0.0, 0.0, 0.0, 0.0, 0.0)
  );
}

TEST(GroupPartition, CollapseTwoGroups) {
  GroupPartition const y(VEC(GroupId, 1, 1, 2, 2));
  GroupPartition const collapsed = y.collapse();

  EXPECT_EQ(collapsed.groups, (std::set<int>{0}));
  EXPECT_EQ(collapsed.subgroups.at(0), (std::set<int>{1, 2}));
}

TEST(GroupPartition, CollapseThreeGroups) {
  GroupPartition const y(VEC(GroupId, 0, 0, 1, 1, 2, 2));
  GroupPartition const collapsed = y.collapse();

  EXPECT_EQ(collapsed.groups, (std::set<int>{0}));
  EXPECT_EQ(collapsed.subgroups.at(0), (std::set<int>{0, 1, 2}));
}

TEST(GroupPartition, CollapseSingleGroup) {
  GroupPartition const y(VEC(GroupId, 0, 0, 0));
  GroupPartition const collapsed = y.collapse();

  EXPECT_EQ(collapsed.groups, (std::set<int>{0}));
  EXPECT_EQ(collapsed.subgroups.at(0), (std::set<int>{0}));
}

TEST(GroupPartition, CollapsePreservesData) {
  FeatureMatrix x = MAT(Feature, rows(6), 2, 2, 2, 4, 4, 4, 1, 1, 1, 6, 6, 6, 3, 3, 3, 5, 5, 5);

  GroupPartition const y(VEC(GroupId, 1, 1, 2, 2, 3, 3));
  GroupPartition collapsed = y.collapse();

  EXPECT_EQ_DATA(collapsed.data(x), x);
}

TEST(GroupPartitionRemapped, Group) {
  FeatureMatrix x = MAT(Feature, rows(6), 2, 2, 2, 4, 4, 4, 1, 1, 1, 6, 6, 6, 3, 3, 3, 5, 5, 5);

  GroupPartition const y(VEC(GroupId, 1, 1, 2, 2, 3, 3));
  GroupPartition const remapped = y.remap({{1, 1}, {2, 1}, {3, 2}});

  EXPECT_EQ_DATA(remapped.group(x, 1), MAT(Feature, rows(4), 2, 2, 2, 4, 4, 4, 1, 1, 1, 6, 6, 6));
  EXPECT_EQ_DATA(remapped.group(x, 2), MAT(Feature, rows(2), 3, 3, 3, 5, 5, 5));
}

TEST(GroupPartitionRemapped, BetweenGroupsSumOfSquaresMultipleGroupsMultivariate) {
  FeatureMatrix x =
      MAT(Feature,
          rows(8),
          23.0,
          1.0,
          1.0,
          25.0,
          1.0,
          1.0,
          18.0,
          1.0,
          1.0,
          29.0,
          1.0,
          1.0,
          19.0,
          1.0,
          1.0,
          21.0,
          1.0,
          1.0,
          35.0,
          1.0,
          1.0,
          17.0,
          1.0,
          1.0);

  GroupPartition const y(VEC(GroupId, 0, 0, 1, 2, 2, 3, 4, 4));
  GroupPartition const remapped = y.remap({{0, 0}, {1, 0}, {2, 1}, {3, 1}, {4, 2}});

  EXPECT_EQ_DATA(remapped.bgss(x), MAT(Feature, rows(3), 19.875, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
}

TEST(GroupPartitionRemapped, WithinGroupsSumOfSquaresMultipleGroupsMultivariate) {
  FeatureMatrix x =
      MAT(Feature,
          rows(8),
          1.0,
          2.0,
          3.0,
          4.0,
          5.0,
          6.0,
          7.0,
          8.0,
          9.0,
          3.0,
          2.0,
          1.0,
          4.0,
          3.0,
          2.0,
          5.0,
          4.0,
          3.0,
          9.0,
          8.0,
          7.0,
          6.0,
          5.0,
          4.0);

  GroupPartition const y(VEC(GroupId, 0, 0, 1, 2, 2, 3, 4, 4));
  GroupPartition const remapped = y.remap({{0, 0}, {1, 0}, {2, 1}, {3, 1}, {4, 2}});

  EXPECT_EQ_DATA(remapped.wgss(x), MAT(Feature, rows(3), 24.5, 24.5, 24.5, 24.5, 24.5, 24.5, 24.5, 24.5, 24.5));
}

TEST(GroupPartitionSplit, AllGroupsGoLeft) {
  GroupPartition const y(VEC(GroupId, 0, 0, 1, 1, 1, 2, 2));

  auto [left, right] = y.split({{0, 2}, {1, 3}, {2, 2}});

  EXPECT_EQ(left.groups, (std::set<int>{0, 1, 2}));
  EXPECT_EQ(left.group_size(0), 2);
  EXPECT_EQ(left.group_size(1), 3);
  EXPECT_EQ(left.group_size(2), 2);
  EXPECT_TRUE(right.groups.empty());
}

TEST(GroupPartitionSplit, AllGroupsGoRight) {
  GroupPartition const y(VEC(GroupId, 0, 0, 1, 1, 1));

  auto [left, right] = y.split({});

  EXPECT_TRUE(left.groups.empty());
  EXPECT_EQ(right.groups, (std::set<int>{0, 1}));
  EXPECT_EQ(right.group_size(0), 2);
  EXPECT_EQ(right.group_size(1), 3);
}

TEST(GroupPartitionSplit, WholeGroupsByGroup) {
  // Equivalent to subset: group 0 goes entirely left, group 1 goes entirely right.
  GroupPartition const y(VEC(GroupId, 0, 0, 1, 1, 1));

  auto [left, right] = y.split({{0, 2}});

  EXPECT_EQ(left.groups, (std::set<int>{0}));
  EXPECT_EQ(left.group_size(0), 2);

  EXPECT_EQ(right.groups, (std::set<int>{1}));
  EXPECT_EQ(right.group_size(1), 3);
}

TEST(GroupPartitionSplit, SplitGroupInHalf) {
  GroupPartition y(VEC(GroupId, 0, 0, 0, 0, 1, 1));

  // Group 0 [0,3]: first 2 rows left, last 2 rows right. Group 1 entirely right.
  auto [left, right] = y.split({{0, 2}});

  EXPECT_EQ(left.groups, (std::set<int>{0}));
  EXPECT_EQ(left.group_start(0), 0);
  EXPECT_EQ(left.group_end(0), 1);
  EXPECT_EQ(left.group_size(0), 2);

  EXPECT_EQ(right.groups, (std::set<int>{0, 1}));
  EXPECT_EQ(right.group_start(0), 2);
  EXPECT_EQ(right.group_end(0), 3);
  EXPECT_EQ(right.group_size(0), 2);
  EXPECT_EQ(right.group_start(1), 4);
  EXPECT_EQ(right.group_end(1), 5);
  EXPECT_EQ(right.group_size(1), 2);
}

TEST(GroupPartitionSplit, MultipleGroupsSplitAtDifferentPoints) {
  GroupPartition y(VEC(GroupId, 0, 0, 0, 1, 1, 1, 2, 2, 2));

  // Group 0: 1 left, 2 right. Group 1: 2 left, 1 right. Group 2: all 3 left.
  auto [left, right] = y.split({{0, 1}, {1, 2}, {2, 3}});

  EXPECT_EQ(left.groups, (std::set<int>{0, 1, 2}));
  EXPECT_EQ(left.group_size(0), 1);
  EXPECT_EQ(left.group_size(1), 2);
  EXPECT_EQ(left.group_size(2), 3);

  EXPECT_EQ(right.groups, (std::set<int>{0, 1}));
  EXPECT_EQ(right.group_size(0), 2);
  EXPECT_EQ(right.group_size(1), 1);
}

TEST(GroupPartitionSplit, BlockBoundariesAreCorrect) {
  GroupPartition y(VEC(GroupId, 0, 0, 0, 0, 1, 1, 1, 1));

  // Group 0 [0,3]: 3 left, 1 right. Group 1 [4,7]: 1 left, 3 right.
  auto [left, right] = y.split({{0, 3}, {1, 1}});

  EXPECT_EQ(left.group_start(0), 0);
  EXPECT_EQ(left.group_end(0), 2);
  EXPECT_EQ(left.group_start(1), 4);
  EXPECT_EQ(left.group_end(1), 4);

  EXPECT_EQ(right.group_start(0), 3);
  EXPECT_EQ(right.group_end(0), 3);
  EXPECT_EQ(right.group_start(1), 5);
  EXPECT_EQ(right.group_end(1), 7);
}

TEST(GroupPartitionSplit, ExtractsCorrectRows) {
  FeatureMatrix x = MAT(Feature, rows(6), 10, 20, 30, 40, 50, 60);

  GroupPartition y(VEC(GroupId, 0, 0, 0, 1, 1, 1));

  // Group 0 [0,2]: 2 left, 1 right. Group 1 [3,5]: 1 left, 2 right.
  auto [left, right] = y.split({{0, 2}, {1, 1}});

  FeatureMatrix left_g0 = left.group(x, 0);
  ASSERT_EQ(left_g0.rows(), 2);
  EXPECT_FLOAT_EQ(left_g0(0, 0), 10);
  EXPECT_FLOAT_EQ(left_g0(1, 0), 20);

  FeatureMatrix left_g1 = left.group(x, 1);
  ASSERT_EQ(left_g1.rows(), 1);
  EXPECT_FLOAT_EQ(left_g1(0, 0), 40);

  FeatureMatrix right_g0 = right.group(x, 0);
  ASSERT_EQ(right_g0.rows(), 1);
  EXPECT_FLOAT_EQ(right_g0(0, 0), 30);

  FeatureMatrix right_g1 = right.group(x, 1);
  ASSERT_EQ(right_g1.rows(), 2);
  EXPECT_FLOAT_EQ(right_g1(0, 0), 50);
  EXPECT_FLOAT_EQ(right_g1(1, 0), 60);
}

TEST(GroupPartitionSplit, InvalidLeftCountThrows) {
  GroupPartition y(VEC(GroupId, 0, 0, 1, 1));

  EXPECT_THROW(y.split({{0, 3}}), std::exception);
  EXPECT_THROW(y.split({{0, -1}}), std::exception);
}

TEST(GroupPartition, RangeCtorSingleGroup) {
  GroupPartition y_part(5, 8);

  EXPECT_EQ(y_part.groups.size(), 1);
  EXPECT_EQ(y_part.group_start(0), 5);
  EXPECT_EQ(y_part.group_end(0), 8);
  EXPECT_EQ(y_part.group_size(0), 4);
}

TEST(GroupPartition, RangeCtorSingleRow) {
  GroupPartition y_part(3, 3);

  EXPECT_EQ(y_part.groups.size(), 1);
  EXPECT_EQ(y_part.group_size(0), 1);
}

TEST(GroupPartition, RangeCtorRejectsNegativeStart) {
  EXPECT_THROW(GroupPartition(-1, 5), std::exception);
}

TEST(GroupPartition, RangeCtorRejectsEndBeforeStart) {
  EXPECT_THROW(GroupPartition(5, 3), std::exception);
}

TEST(GroupPartition, BisectStartingAtZero) {
  GroupPartition y_part = GroupPartition(0, 5).bisect(3);

  EXPECT_EQ(y_part.groups.size(), 2);
  EXPECT_EQ(y_part.group_start(0), 0);
  EXPECT_EQ(y_part.group_end(0), 2);
  EXPECT_EQ(y_part.group_size(0), 3);
  EXPECT_EQ(y_part.group_start(1), 3);
  EXPECT_EQ(y_part.group_end(1), 5);
  EXPECT_EQ(y_part.group_size(1), 3);
}

TEST(GroupPartition, BisectAtOffset) {
  GroupPartition y_part = GroupPartition(10, 19).bisect(15);

  EXPECT_EQ(y_part.groups.size(), 2);
  EXPECT_EQ(y_part.group_start(0), 10);
  EXPECT_EQ(y_part.group_end(0), 14);
  EXPECT_EQ(y_part.group_size(0), 5);
  EXPECT_EQ(y_part.group_start(1), 15);
  EXPECT_EQ(y_part.group_end(1), 19);
  EXPECT_EQ(y_part.group_size(1), 5);
}

TEST(GroupPartition, BisectMinimal) {
  // Smallest valid split: one row per group.
  GroupPartition y_part = GroupPartition(0, 1).bisect(1);

  EXPECT_EQ(y_part.groups.size(), 2);
  EXPECT_EQ(y_part.group_size(0), 1);
  EXPECT_EQ(y_part.group_size(1), 1);
}

TEST(GroupPartition, BisectRejectsMidAtOrBeforeStart) {
  // mid must be strictly greater than start (group 0 must have ≥ 1 row).
  EXPECT_THROW(GroupPartition(0, 5).bisect(0), std::exception);
  EXPECT_THROW(GroupPartition(5, 9).bisect(3), std::exception);
}

TEST(GroupPartition, BisectRejectsMidBeyondEnd) {
  // mid must be ≤ end (group 1 must have ≥ 1 row).
  EXPECT_THROW(GroupPartition(0, 5).bisect(6), std::exception);
}

TEST(GroupPartition, BisectRejectsMultiGroupReceiver) {
  // Receiver must be a single-group partition.
  GroupPartition two = GroupPartition(0, 5).bisect(3);
  EXPECT_THROW(two.bisect(2), std::exception);
}
