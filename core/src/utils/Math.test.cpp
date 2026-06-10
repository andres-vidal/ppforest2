#include <gtest/gtest.h>

#include "utils/Math.hpp"

using namespace ppforest2;
using namespace ppforest2::types;
using namespace ppforest2::math;

TEST(Collinear, CollinearSameDirection) {
  Vector<float> a(3);
  a << 1.0, 2.0, 6.0;

  Vector<float> b(3);
  b << 2.0, 4.0, 12.0;

  ASSERT_TRUE(collinear(a, b));
}

TEST(Collinear, CollinearOppositeDirection) {
  Vector<float> a(3);
  a << 1.0, 2.0, 6.0;

  Vector<float> b(3);
  b << -1.0, -2.0, -6.0;

  ASSERT_TRUE(collinear(a, b));
}

TEST(Collinear, NonCollinear) {
  Vector<float> a(3);
  a << 0.0, 1.0, 0.0;

  Vector<float> b(3);
  b << 1.0, 0.0, 0.0;

  ASSERT_FALSE(collinear(a, b));
}

TEST(Collinear, HighDimensional) {
  Vector<float> a(5);
  a << 1.0, 2.0, 3.0, 4.0, 5.0;

  Vector<float> b(5);
  b << 3.0, 6.0, 9.0, 12.0, 15.0;

  ASSERT_TRUE(collinear(a, b));
}

TEST(Collinear, NearlyCollinear) {
  Vector<float> a(3);
  a << 1.0, 2.0, 3.0;

  Vector<float> b(3);
  b << 1.001, 2.002, 3.003;

  ASSERT_TRUE(collinear(a, b));
}

TEST(Collinear, SlightlyNonCollinear) {
  Vector<float> a(3);
  a << 1.0, 0.0, 0.0;

  Vector<float> b(3);
  b << 1.0, 0.5, 0.0;

  ASSERT_FALSE(collinear(a, b));
}

TEST(Collinear, UnitVectors) {
  Vector<float> a(2);
  a << 1.0, 0.0;

  Vector<float> b(2);
  b << -1.0, 0.0;

  ASSERT_TRUE(collinear(a, b));
}

TEST(IsApprox, WithinThreshold) {
  ASSERT_TRUE(is_approx(1.0, 1.005, 0.01));
}

TEST(IsApprox, OutsideThreshold) {
  ASSERT_FALSE(is_approx(1.0, 1.02, 0.01));
}

TEST(IsApprox, DefaultThreshold) {
  ASSERT_TRUE(is_approx(1.0, 1.005));
  ASSERT_FALSE(is_approx(1.0, 1.02));
}

TEST(IsModuleApprox, SameSign) {
  ASSERT_TRUE(is_module_approx(5.0, 5.005));
}

TEST(IsModuleApprox, OppositeSign) {
  ASSERT_TRUE(is_module_approx(5.0, -5.005));
}

TEST(IsModuleApprox, Different) {
  ASSERT_FALSE(is_module_approx(5.0, 6.0));
}

TEST(ProportionToCount, RoundsHalfToEven) {
  EXPECT_EQ(proportion_to_count(0.5F, 4), 2);  // 2.0
  EXPECT_EQ(proportion_to_count(0.5F, 5), 2);  // 2.5 -> 2 (even)
  EXPECT_EQ(proportion_to_count(0.5F, 7), 4);  // 3.5 -> 4 (even)
  EXPECT_EQ(proportion_to_count(0.5F, 9), 4);  // 4.5 -> 4 (even)
}

TEST(ProportionToCount, ClampsToAtLeastOne) {
  EXPECT_EQ(proportion_to_count(0.1F, 4), 1);  // 0.4 -> 0 -> clamped to 1
  EXPECT_EQ(proportion_to_count(0.5F, 1), 1);  // 0.5 -> 0 -> clamped to 1
}

TEST(ProportionToCount, FullProportionSelectsAll) {
  EXPECT_EQ(proportion_to_count(1.0F, 10), 10);
}
