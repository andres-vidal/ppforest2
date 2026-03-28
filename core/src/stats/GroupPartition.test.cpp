#include <gtest/gtest.h>

#include "stats/GroupPartition.hpp"
#include "utils/Types.hpp"

#include "utils/Macros.hpp"

using namespace ppforest2;
using namespace ppforest2::stats;
using namespace ppforest2::types;

TEST(GroupPartition, GroupSize) {
  FeatureMatrix x = MAT(Feature, rows(6),
      2, 2, 2,
      4, 4, 4,
      1, 1, 1,
      6, 6, 6,
      3, 3, 3,
      5, 5, 5);

  ResponseVector y = VEC(Response,
      1,
      1,
      2,
      2,
      3,
      3);

  GroupPartition spec(y);

  ASSERT_EQ(2, spec.group_size(1));
  ASSERT_EQ(2, spec.group_size(2));
  ASSERT_EQ(2, spec.group_size(3));
}

TEST(GroupPartition, GroupStart) {
  FeatureMatrix x = MAT(Feature, rows(6),
      2, 2, 2,
      4, 4, 4,
      1, 1, 1,
      6, 6, 6,
      3, 3, 3,
      5, 5, 5);

  ResponseVector y = VEC(Response,
      1,
      1,
      2,
      2,
      3,
      3);

  GroupPartition spec(y);

  ASSERT_EQ(0, spec.group_start(1));
  ASSERT_EQ(2, spec.group_start(2));
  ASSERT_EQ(4, spec.group_start(3));
}

TEST(GroupPartition, GroupEnd) {
  FeatureMatrix x = MAT(Feature, rows(6),
      2, 2, 2,
      4, 4, 4,
      1, 1, 1,
      6, 6, 6,
      3, 3, 3,
      5, 5, 5);

  ResponseVector y = VEC(Response,
      1,
      1,
      2,
      2,
      3,
      3);

  GroupPartition spec(y);

  ASSERT_EQ(1, spec.group_end(1));
  ASSERT_EQ(3, spec.group_end(2));
  ASSERT_EQ(5, spec.group_end(3));
}

TEST(GroupPartition, Group) {
  FeatureMatrix x = MAT(Feature, rows(6),
      2, 2, 2,
      4, 4, 4,
      1, 1, 1,
      6, 6, 6,
      3, 3, 3,
      5, 5, 5);


  ResponseVector y = VEC(Response,
      1,
      1,
      2,
      2,
      3,
      3);

  GroupPartition spec(y);

  FeatureMatrix actual = spec.group(x, 1);

  FeatureMatrix expected = MAT(Feature, rows(2),
      2, 2, 2,
      4, 4, 4);

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);

  actual = spec.group(x, 2);

  expected = MAT(Feature, rows(2),
      1, 1, 1,
      6, 6, 6);

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);

  actual = spec.group(x, 3);

  expected = MAT(Feature, rows(2),
      3, 3, 3,
      5, 5, 5);

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(GroupPartition, ErrorGroupsNotContiguous) {
  FeatureMatrix x = MAT(Feature, rows(6),
      2, 2, 2,
      4, 4, 4,
      1, 1, 1,
      6, 6, 6,
      3, 3, 3,
      5, 5, 5);

  ResponseVector y = VEC(Response,
      1,
      2,
      3,
      1,
      2,
      3);

  ASSERT_THROW((GroupPartition(y)), std::invalid_argument);
}

TEST(GroupPartition, Subset) {
  FeatureMatrix x = MAT(Feature, rows(6),
      1, 2, 2,
      1, 4, 4,
      2, 1, 1,
      2, 6, 6,
      3, 3, 3,
      3, 5, 5);

  ResponseVector y = VEC(Response,
      1,
      1,
      2,
      2,
      3,
      3);

  GroupPartition spec = GroupPartition(y).subset({ 1, 3 });

  ASSERT_EQ(2, spec.group_size(1));
  ASSERT_EQ(0, spec.group_start(1));
  ASSERT_EQ(1, spec.group_end(1));

  ASSERT_EQ(2, spec.group_size(3));
  ASSERT_EQ(4, spec.group_start(3));
  ASSERT_EQ(5, spec.group_end(3));

  FeatureMatrix expected_group_1 = MAT(Feature, rows(2),
      1, 2, 2,
      1, 4, 4);

  ASSERT_EQ_DATA(expected_group_1, spec.group(x, 1));

  FeatureMatrix expected_group_3 = MAT(Feature, rows(2),
      3, 3, 3,
      3, 5, 5);

  FeatureMatrix expected_x = MAT(Feature, rows(4),
      1, 2, 2,
      1, 4, 4,
      3, 3, 3,
      3, 5, 5);

  ASSERT_EQ_DATA(expected_x, spec.data(x));

  FeatureVector expected_mean = VEC(Feature,
      2.0,
      3.5,
      3.5);

  ASSERT_EQ(expected_mean, spec.mean(x));
}

TEST(GroupPartition, Remap) {
  FeatureMatrix x = MAT(Feature, rows(6),
      1, 1, 1,
      1, 2, 2,
      2, 1, 1,
      2, 2, 2,
      3, 1, 1,
      3, 2, 2);

  ResponseVector y = VEC(Response,
      1,
      1,
      2,
      2,
      3,
      3);

  GroupPartition spec(y);

  std::map<int, int> mapping = {
    { 1, 0 },
    { 2, 1 },
    { 3, 0 }
  };

  GroupPartition remapped = spec.remap(mapping);

  FeatureMatrix remapped_x = remapped.data(x);

  ASSERT_EQ_DATA(x, remapped_x);
  ASSERT_EQ(std::set<int>({ 0, 1 }), remapped.groups);
}

TEST(GroupPartition,  BetweenGroupsSumOfSquaresSingleGroup) {
  FeatureMatrix x = MAT(Feature, rows(3),
      1.0, 2.0, 6.0,
      2.0, 3.0, 7.0,
      3.0, 4.0, 8.0);

  ResponseVector y = VEC(Response,
      0,
      0,
      0);

  GroupPartition spec(y);

  FeatureMatrix actual = spec.bgss(x);

  FeatureMatrix expected = MAT(Feature, rows(3),
      0.0, 0.0, 0.0,
      0.0, 0.0, 0.0,
      0.0, 0.0, 0.0);

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(GroupPartition,  BetweenGroupsSumOfSquaresTwoEqualGroups) {
  FeatureMatrix x = MAT(Feature, rows(6),
      1.0, 2.0, 6.0,
      2.0, 3.0, 7.0,
      3.0, 4.0, 8.0,
      1.0, 2.0, 6.0,
      2.0, 3.0, 7.0,
      3.0, 4.0, 8.0);

  ResponseVector y = VEC(Response,
      0,
      0,
      0,
      1,
      1,
      1);

  GroupPartition spec(y);

  FeatureMatrix actual = spec.bgss(x);

  FeatureMatrix expected = MAT(Feature, rows(3),
      0.0, 0.0, 0.0,
      0.0, 0.0, 0.0,
      0.0, 0.0, 0.0);


  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(GroupPartition,  BetweenGroupsSumOfSquaresMultipleGroupsUnivariate) {
  FeatureMatrix x = MAT(Feature, rows(8),
      23.0,
      25.0,
      18.0,
      29.0,
      19.0,
      21.0,
      35.0,
      17.0);

  ResponseVector y = VEC(Response,
      0,
      0,
      0,
      1,
      1,
      1,
      2,
      2);

  GroupPartition spec(y);

  FeatureMatrix actual = spec.bgss(x);

  FeatureMatrix expected = MAT(Feature, rows(1),
      19.875);

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(GroupPartition,  BetweenGroupsSumOfSquaresMultipleGroupsUnivariateNonSequentialGroups) {
  FeatureMatrix x = MAT(Feature, rows(8),
      23.0,
      25.0,
      18.0,
      29.0,
      19.0,
      21.0,
      35.0,
      17.0);

  ResponseVector y = VEC(Response,
      1,
      1,
      1,
      7,
      7,
      7,
      3,
      3);

  GroupPartition spec(y);

  FeatureMatrix actual = spec.bgss(x);

  FeatureMatrix expected = MAT(Feature, rows(1),
      19.875);

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(GroupPartition,  BetweenGroupsSumOfSquaresMultipleGroupsMultivariate) {
  FeatureMatrix x = MAT(Feature, rows(8),
      23.0, 1.0, 1.0,
      25.0, 1.0, 1.0,
      18.0, 1.0, 1.0,
      29.0, 1.0, 1.0,
      19.0, 1.0, 1.0,
      21.0, 1.0, 1.0,
      35.0, 1.0, 1.0,
      17.0, 1.0, 1.0);

  ResponseVector y = VEC(Response,
      0,
      0,
      0,
      1,
      1,
      1,
      2,
      2);

  GroupPartition spec(y);

  FeatureMatrix actual = spec.bgss(x);

  FeatureMatrix expected = MAT(Feature, rows(3),
      19.875, 0.0, 0.0,
      0.0,    0.0, 0.0,
      0.0,    0.0, 0.0);

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(GroupPartition,  WithinGroupsSumOfSquaresSingleGroupNoVariance) {
  FeatureMatrix x = MAT(Feature, rows(3),
      1.0, 1.0, 1.0,
      1.0, 1.0, 1.0,
      1.0, 1.0, 1.0);

  ResponseVector y = VEC(Response,
      0,
      0,
      0);

  GroupPartition spec(y);

  FeatureMatrix actual = spec.wgss(x);

  FeatureMatrix expected = MAT(Feature, rows(3),
      0.0, 0.0, 0.0,
      0.0, 0.0, 0.0,
      0.0, 0.0, 0.0);

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(GroupPartition,  WithinGroupsSumOfSquaresSingleGroupWithVariance) {
  FeatureMatrix x = MAT(Feature, rows(3),
      1.0, 1.0, 1.0,
      2.0, 2.0, 2.0,
      3.0, 3.0, 3.0);

  ResponseVector y = VEC(Response,
      0,
      0,
      0);

  GroupPartition spec(y);

  FeatureMatrix actual = spec.wgss(x);

  FeatureMatrix expected = MAT(Feature, rows(3),
      2.0, 2.0, 2.0,
      2.0, 2.0, 2.0,
      2.0, 2.0, 2.0);

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(GroupPartition,  WithinGroupsSumOfSquaresTwoEqualGroups) {
  FeatureMatrix x = MAT(Feature, rows(6),
      1.0, 1.0, 1.0,
      2.0, 2.0, 2.0,
      3.0, 3.0, 3.0,
      1.0, 1.0, 1.0,
      2.0, 2.0, 2.0,
      3.0, 3.0, 3.0);

  ResponseVector y = VEC(Response,
      0,
      0,
      0,
      1,
      1,
      1);

  GroupPartition spec(y);

  FeatureMatrix actual = spec.wgss(x);

  FeatureMatrix expected = MAT(Feature, rows(3),
      4.0, 4.0, 4.0,
      4.0, 4.0, 4.0,
      4.0, 4.0, 4.0);

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(GroupPartition,  WithinGroupsSumOfSquaresTwoGroupsSameVariance) {
  FeatureMatrix x = MAT(Feature, rows(6),
      1.0, 1.0, 1.0,
      2.0, 2.0, 2.0,
      3.0, 3.0, 3.0,
      4.0, 4.0, 4.0,
      5.0, 5.0, 5.0,
      6.0, 6.0, 6.0);

  ResponseVector y = VEC(Response,
      0,
      0,
      0,
      1,
      1,
      1);

  GroupPartition spec(y);

  FeatureMatrix actual = spec.wgss(x);

  FeatureMatrix expected = MAT(Feature, rows(3),
      4.0, 4.0, 4.0,
      4.0, 4.0, 4.0,
      4.0, 4.0, 4.0);

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(GroupPartition,  WithinGroupsSumOfSquaresTwoGroupsDifferentVariance) {
  FeatureMatrix x = MAT(Feature, rows(6),
      1.0, 1.0, 1.0,
      2.0, 2.0, 2.0,
      3.0, 3.0, 3.0,
      1.0, 1.0, 1.0,
      5.0, 5.0, 5.0,
      6.0, 6.0, 6.0);

  ResponseVector y = VEC(Response,
      0,
      0,
      0,
      1,
      1,
      1);

  GroupPartition spec(y);

  FeatureMatrix actual = spec.wgss(x);

  FeatureMatrix expected = MAT(Feature, rows(3),
      16.0, 16.0, 16.0,
      16.0, 16.0, 16.0,
      16.0, 16.0, 16.0);

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(GroupPartition,  WithinGroupsSumOfSquaresMultipleGroupsMultivariate1) {
  FeatureMatrix x = MAT(Feature, rows(8),
      1.0, 2.0, 3.0,
      4.0, 5.0, 6.0,
      7.0, 8.0, 9.0,
      3.0, 2.0, 1.0,
      4.0, 3.0, 2.0,
      5.0, 4.0, 3.0,
      9.0, 8.0, 7.0,
      6.0, 5.0, 4.0);

  ResponseVector y = VEC(Response,
      0,
      0,
      0,
      1,
      1,
      1,
      2,
      2);


  GroupPartition spec(y);

  FeatureMatrix actual = spec.wgss(x);

  FeatureMatrix expected = MAT(Feature, rows(3),
      24.5, 24.5, 24.5,
      24.5, 24.5, 24.5,
      24.5, 24.5, 24.5);

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(GroupPartition,  WithinGroupsSumOfSquaresMultipleGroupsMultivariate2) {
  FeatureMatrix x = MAT(Feature, rows(8),
      1.0, 2.0, 3.0, 0.0,
      4.0, 5.0, 6.0, 0.0,
      7.0, 8.0, 9.0, 0.0,
      3.0, 2.0, 1.0, 0.0,
      4.0, 3.0, 2.0, 0.0,
      5.0, 4.0, 3.0, 0.0,
      9.0, 8.0, 7.0, 0.0,
      6.0, 5.0, 4.0, 0.0);

  ResponseVector y = VEC(Response,
      0,
      0,
      0,
      1,
      1,
      1,
      2,
      2);

  GroupPartition spec(y);

  FeatureMatrix actual = spec.wgss(x);

  FeatureMatrix expected = MAT(Feature, rows(4),
      24.5, 24.5, 24.5, 0.0,
      24.5, 24.5, 24.5, 0.0,
      24.5, 24.5, 24.5, 0.0,
      0.0,  0.0,  0.0,  0.0);

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(GroupPartition, CollapseTwoGroups) {
  ResponseVector y = VEC(Response,
      1,
      1,
      2,
      2);

  GroupPartition spec(y);
  GroupPartition collapsed = spec.collapse();

  ASSERT_EQ(1u, collapsed.groups.size());
  ASSERT_EQ(std::set<int>({ 0 }), collapsed.groups);
}

TEST(GroupPartition, CollapseThreeGroups) {
  ResponseVector y = VEC(Response,
      0,
      0,
      1,
      1,
      2,
      2);

  GroupPartition spec(y);
  GroupPartition collapsed = spec.collapse();

  ASSERT_EQ(1u, collapsed.groups.size());
  ASSERT_EQ(std::set<int>({ 0 }), collapsed.groups);
  ASSERT_EQ(std::set<int>({ 0, 1, 2 }), collapsed.subgroups.at(0));
}

TEST(GroupPartition, CollapseSingleGroup) {
  ResponseVector y = VEC(Response,
      0,
      0,
      0);

  GroupPartition spec(y);
  GroupPartition collapsed = spec.collapse();

  ASSERT_EQ(1u, collapsed.groups.size());
  ASSERT_EQ(std::set<int>({ 0 }), collapsed.groups);
}

TEST(GroupPartition, CollapsePreservesData) {
  FeatureMatrix x = MAT(Feature, rows(6),
      2, 2, 2,
      4, 4, 4,
      1, 1, 1,
      6, 6, 6,
      3, 3, 3,
      5, 5, 5);

  ResponseVector y = VEC(Response,
      1,
      1,
      2,
      2,
      3,
      3);

  GroupPartition spec(y);
  GroupPartition collapsed = spec.collapse();

  ASSERT_EQ_DATA(x, collapsed.data(x));
}

TEST(GroupPartitionRemapped, Group) {
  FeatureMatrix x = MAT(Feature, rows(6),
      2, 2, 2,
      4, 4, 4,
      1, 1, 1,
      6, 6, 6,
      3, 3, 3,
      5, 5, 5);


  ResponseVector y = VEC(Response,
      1,
      1,
      2,
      2,
      3,
      3);

  GroupPartition base(y);
  GroupPartition spec = base.remap({ { 1, 1 }, { 2, 1 }, { 3, 2 } });

  FeatureMatrix actual = spec.group(x, 1);

  FeatureMatrix expected = MAT(Feature, rows(4),
      2, 2, 2,
      4, 4, 4,
      1, 1, 1,
      6, 6, 6);

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);

  actual = spec.group(x, 2);

  expected = MAT(Feature, rows(2),
      3, 3, 3,
      5, 5, 5);

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(GroupPartitionRemapped, BetweenGroupsSumOfSquaresMultipleGroupsMultivariate) {
  FeatureMatrix x = MAT(Feature, rows(8),
      23.0, 1.0, 1.0,
      25.0, 1.0, 1.0,
      18.0, 1.0, 1.0,
      29.0, 1.0, 1.0,
      19.0, 1.0, 1.0,
      21.0, 1.0, 1.0,
      35.0, 1.0, 1.0,
      17.0, 1.0, 1.0);

  ResponseVector y = VEC(Response,
      0,
      0,
      1,
      2,
      2,
      3,
      4,
      4);

  GroupPartition spec(y);
  GroupPartition remapped = spec.remap({ { 0, 0 }, { 1, 0 }, { 2, 1 }, { 3, 1 }, { 4, 2 } });

  FeatureMatrix actual = remapped.bgss(x);

  FeatureMatrix expected = MAT(Feature, rows(3),
      19.875, 0.0, 0.0,
      0.0,    0.0, 0.0,
      0.0,    0.0, 0.0);

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}

TEST(GroupPartitionRemapped, WithinGroupsSumOfSquaresMultipleGroupsMultivariate) {
  FeatureMatrix x = MAT(Feature, rows(8),
      1.0, 2.0, 3.0,
      4.0, 5.0, 6.0,
      7.0, 8.0, 9.0,
      3.0, 2.0, 1.0,
      4.0, 3.0, 2.0,
      5.0, 4.0, 3.0,
      9.0, 8.0, 7.0,
      6.0, 5.0, 4.0);

  ResponseVector y = VEC(Response,
      0,
      0,
      1,
      2,
      2,
      3,
      4,
      4);


  GroupPartition spec(y);

  GroupPartition remapped = spec.remap({ { 0, 0 }, { 1, 0 }, { 2, 1 }, { 3, 1 }, { 4, 2 } });

  FeatureMatrix actual = remapped.wgss(x);

  FeatureMatrix expected = MAT(Feature, rows(3),
      24.5, 24.5, 24.5,
      24.5, 24.5, 24.5,
      24.5, 24.5, 24.5);

  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.rows(), actual.rows());
  ASSERT_EQ(expected.cols(), actual.cols());
  ASSERT_EQ(expected, actual);
}
