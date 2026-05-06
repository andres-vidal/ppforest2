#pragma once

#include <cstddef>
#include <numeric>
#include <vector>

namespace ppforest2::utils {
  /**
   * @brief Build the sequence `[0, 1, ..., n - 1]` as `std::vector<int>`.
   *
   * Replaces the pattern
   * @code
   *   std::vector<int> v(n);
   *   std::iota(v.begin(), v.end(), 0);
   * @endcode
   * which appears anywhere we need a permutation buffer or a "select all
   * rows/columns" index list. `int` is the right element type for every
   * caller in the codebase: Eigen's indexed assignment, Eigen column
   * selection, and the strategy `Result` types all consume `int` indices.
   *
   * @tparam Size  Any integral type (e.g. `int`, `std::size_t`,
   *               `Eigen::Index`). Cast to `std::size_t` for the
   *               `std::vector` constructor.
   * @param  n     Length of the resulting sequence (must be `>= 0`).
   */
  template<typename Size> inline std::vector<int> range_vector(Size n) {
    std::vector<int> result(static_cast<std::size_t>(n));
    std::iota(result.begin(), result.end(), 0);
    return result;
  }
}
