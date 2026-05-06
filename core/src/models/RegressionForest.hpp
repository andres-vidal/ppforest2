#pragma once

#include "models/Forest.hpp"

namespace ppforest2 {
  /**
   * @brief Random forest of regression trees.
   *
   * Aggregates per-tree predictions by arithmetic mean. Use the free
   * functions in `Metrics.hpp` for OOB diagnostics and variable
   * importance.
   */
  class RegressionForest : public Forest {
  public:
    using Ptr = std::unique_ptr<RegressionForest>;
    using Forest::predict;

    using FeatureMatrix = types::FeatureMatrix;
    using FeatureVector = types::FeatureVector;
    using OutcomeVector = types::OutcomeVector;
    using Outcome       = types::Outcome;


    RegressionForest();
    explicit RegressionForest(TrainingSpec::Ptr spec);

    static Ptr train(TrainingSpec const& spec, FeatureMatrix const& x, OutcomeVector const& y);

    types::Outcome predict(FeatureVector const& x) const override;

    void accept(Model::Visitor& visitor) const override;

  protected:
    BaggedTree::Ptr train_tree(FeatureMatrix const& x, OutcomeVector const& y, stats::RNG& rng) const override;
  };
}
