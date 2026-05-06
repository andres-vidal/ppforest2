#include "utils/RangeVector.hpp"

#include <gtest/gtest.h>

#include <cstddef>

using namespace ppforest2::utils;

TEST(RangeVector, EmptyForZero) {
  EXPECT_TRUE(range_vector(0).empty());
}

TEST(RangeVector, AscendingFromZero) {
  std::vector<int> const v = range_vector(5);
  ASSERT_EQ(v.size(), 5U);
  EXPECT_EQ(v[0], 0);
  EXPECT_EQ(v[1], 1);
  EXPECT_EQ(v[2], 2);
  EXPECT_EQ(v[3], 3);
  EXPECT_EQ(v[4], 4);
}

TEST(RangeVector, AcceptsSizeT) {
  std::size_t const n      = 3;
  std::vector<int> const v = range_vector(n);
  ASSERT_EQ(v.size(), 3U);
  EXPECT_EQ(v.back(), 2);
}

TEST(RangeVector, ResultIsIntVector) {
  static_assert(std::is_same_v<decltype(range_vector(1)), std::vector<int>>);
}
