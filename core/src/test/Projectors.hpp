#pragma once

#include "models/Projector.hpp"
#include "utils/Types.hpp"

#include <Eigen/Dense>
#include <vector>

namespace ppforest2::test {
  /**
   * @brief Build a `Projector` from a `std::vector<Feature>` initializer.
   *
   * Tests reach for projector vectors a lot — the `Eigen::Map → Projector`
   * dance reads as line noise at the call site. This helper hides it so
   * tests can write `as_projector({1.0F, 0.0F})` and stay focused on the
   * data their assertions depend on.
   */
  inline pp::Projector as_projector(std::vector<types::Feature> v) {
    return Eigen::Map<pp::Projector>(v.data(), v.size());
  }
}
