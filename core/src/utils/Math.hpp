#pragma once

#include <algorithm>
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

  /**
   * @brief Convert a proportion of a total into a count.
   *
   * Turns a feature proportion (e.g. `p_vars = 0.5`) into a variable-selection
   * count, rounding half to even (banker's rounding) to match R's `round()`,
   * then clamping to at least 1 so a tiny proportion still selects one item
   * (matching R's `max(1L, ...)`). Shared by the CLI and the R package so both
   * resolve identical counts for the same proportion.
   *
   * Half-to-even is computed by hand rather than via `std::nearbyint` so the
   * result never depends on the current floating-point rounding mode.
   * `std::floor`, the subtraction, and integer parity are all mode-independent
   * and exact for these inputs, keeping results reproducible across compilers
   * and platforms.
   *
   * @param p      Proportion in (0, 1].
   * @param total  Total number of items (>= 1).
   * @return       Count in [1, total].
   */
  inline int proportion_to_count(float p, unsigned int total) {
    float const value = static_cast<float>(total) * p;
    float const lower = std::floor(value);
    float const fract = value - lower;

    int count = static_cast<int>(lower);
    // Round up past the halfway point, or exactly at it when the floor is odd
    // (ties to even). Mirrors std::nearbyint's default-mode result without
    // reading the floating-point environment.
    if (fract > 0.5F || (fract == 0.5F && count % 2 != 0)) {
      count += 1;
    }
    return std::max(1, count);
  }
}
