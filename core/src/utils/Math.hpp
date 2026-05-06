#pragma once

#include <cmath>

#include "utils/Types.hpp"

#define APPROX_THRESHOLD 0.01

/** @brief Numeric comparison utilities. */
namespace ppforest2::math {
  /**
   * @brief Check whether two scalars are approximately equal.
   *
   * @param a          First value.
   * @param b          Second value.
   * @param threshold  Maximum allowed absolute difference.
   * @return           True if |a − b| < threshold.
   */
  template<typename A, typename B, typename T> inline bool is_approx(A a, B b, T threshold) {
    return fabs(a - b) < threshold;
  }

  /** @brief Overload using the default APPROX_THRESHOLD. */
  template<typename A, typename B> inline bool is_approx(A a, B b) {
    return is_approx(a, b, APPROX_THRESHOLD);
  }

  /**
   * @brief Check whether the absolute values of two scalars are approximately equal.
   */
  template<typename A, typename B> inline bool is_module_approx(A a, B b) {
    return is_approx(fabs(a), fabs(b));
  }

  /**
   * @brief Check whether two vectors are collinear (parallel or anti-parallel).
   *
   * @param a  First vector.
   * @param b  Second vector (same dimension as @p a).
   * @return   True if |cos(angle)| ≈ 1.
   */
  template<typename T> bool collinear(types::Vector<T> const& a, types::Vector<T> const& b) {
    return is_module_approx(a.dot(b) / (a.norm() * b.norm()), 1.0);
  }
}
