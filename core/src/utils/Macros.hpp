#pragma once

/**
 * @brief Test helper macros for Eigen comparisons and literal construction.
 *
 * Provides GTest assertion macros for approximate and collinear
 * comparisons, plus VEC/MAT macros that replace Eigen's comma
 * initializer to work around a GCC miscompilation issue.
 */

#include "utils/Invariant.hpp" // IWYU pragma: keep
#include "utils/Math.hpp"
#include <Eigen/Dense>

/** @brief Assert that two Eigen objects are approximately equal. */
#define ASSERT_APPROX(a, b)                                                             \
  ASSERT_TRUE(a.isApprox(b, APPROX_THRESHOLD)) << "Expected " << std::endl              \
                                               << a << std::endl                        \
                                               << " to be approximate to " << std::endl \
                                               << b

/** @brief Assert that two Eigen vectors are collinear (parallel or anti-parallel). */
#define ASSERT_COLLINEAR(a, b)                                                                                    \
  ASSERT_TRUE(ppforest2::math::collinear(a, b)) << "Expected columns of " << std::endl                            \
                                                << a << std::endl                                                 \
                                                << " to be collinear with its respective column of " << std::endl \
                                                << b

/** @brief Assert that two Eigen objects have identical size and values. */
#define ASSERT_EQ_DATA(a, b)     \
  ASSERT_EQ(a.size(), b.size()); \
  ASSERT_EQ(a.rows(), b.rows()); \
  ASSERT_EQ(a.cols(), b.cols()); \
  ASSERT_EQ(a, b);

#define EXPECT_EQ_DATA(a, b)     \
  EXPECT_EQ(a.size(), b.size()); \
  EXPECT_EQ(a.rows(), b.rows()); \
  EXPECT_EQ(a.cols(), b.cols()); \
  EXPECT_EQ(a, b);

// Workaround for GCC miscompilation of long overloaded operator, chains
// (see https://stackoverflow.com/questions/79872387).
// Use these instead of Eigen's comma initializer (operator<<) for matrices and vectors.

constexpr int rows(int n) {
  return n;
}

#define VEC(T, ...)                                                                           \
  ([&]() -> Eigen::Matrix<T, Eigen::Dynamic, 1> {                                             \
    std::vector<T> _vals = {__VA_ARGS__};                                                     \
    return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(_vals.data(), _vals.size()); \
  })()

#define MAT(T, ROWS, ...)                                                                       \
  ([&]() -> Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> {                                  \
    const int _rows      = static_cast<int>(ROWS);                                              \
    std::vector<T> _vals = {__VA_ARGS__};                                                       \
    invariant(_rows > 0, "ROWS must be > 0");                                                   \
    invariant(_vals.size() % _rows == 0, "Element count not divisible by ROWS");                \
    const int _cols = static_cast<int>(_vals.size()) / _rows;                                   \
    return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>( \
        _vals.data(), _rows, _cols                                                              \
    );                                                                                          \
  })()
