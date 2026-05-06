#pragma once

/**
 * @file TestSpec.hpp
 * @brief Minimal mode-correct `TrainingSpec::Ptr` for structural tests.
 *
 * Used by tests that build a `Tree` or `Forest` directly from a hand-built
 * `TreeNode` graph and just need to satisfy the subclass/mode invariants on
 * `ClassificationTree`, `RegressionTree`, etc. Strategy choices are
 * irrelevant — the spec is never trained against.
 *
 * Tests that actually train should construct their `TrainingSpec` inline
 * via `TrainingSpec::builder(...)`; the kind of model being trained is part
 * of what the test is documenting.
 */

#include "models/TrainingSpec.hpp"

namespace ppforest2::test {
  /** @brief Default-valued classification spec. */
  inline TrainingSpec::Ptr classification_spec() {
    return TrainingSpec::builder(types::Mode::Classification).make();
  }

  /**
   * @brief Minimal well-formed regression spec.
   *
   * Picks regression-compatible strategies (`ByCutpoint`, `MeanResponse`,
   * `MinSize`) since the classification defaults (e.g., `PureNode`) reject
   * regression mode at validation time.
   */
  inline TrainingSpec::Ptr regression_spec() {
    return TrainingSpec::builder(types::Mode::Regression)
        .grouping(grouping::by_cutpoint())
        .leaf(leaf::mean_response())
        .stop(stop::min_size(5))
        .make();
  }
}
