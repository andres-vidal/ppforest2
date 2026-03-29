#pragma once

#include "models/Model.hpp"
#include "models/BootstrapTree.hpp"

#include <map>
#include <memory>
#include <numeric>
#include <thread>
#include <vector>

namespace ppforest2 {
  /**
   * @brief A projection pursuit random forest.
   *
   * An ensemble of BootstrapTree instances, each trained on a
   * bootstrap sample.  Predictions are made by majority vote.
   * Out-of-bag estimation and vote-proportion predictions are
   * supported.
   *
   * @code
   *   TrainingSpecUPDA spec(n_vars: 3, lambda: 0.0);
   *   Forest forest = Forest::train(spec, x, y, size: 500, seed: 42);
   *
   *   types::ResponseVector preds = forest.predict(x_test);
   *   double oob = forest.oob_error(x, y);
   *
   *   // Vote proportions — (n × G) matrix, rows sum to 1.
   *   types::FeatureMatrix probs = forest.predict(x_test, Proportions{});
   * @endcode
   */
  struct Forest : public Model {
    /**
     * @brief Train a random forest.
     *
     * @param training_spec  Training specification.
     * @param x              Feature matrix (n × p).
     * @param y              Response vector (n).
     * @param size           Number of trees.
     * @param seed            RNG seed.
     * @param n_threads      Number of threads (default: hardware concurrency).
     * @param max_retries    Maximum retry attempts for degenerate trees (default: 3).
     * @return               Trained forest.
     */
    static Forest train(
      const TrainingSpec&          training_spec,
      const types::FeatureMatrix&  x,
      const types::ResponseVector& y,
      int                          size,
      int                          seed,
      int                          n_threads   = std::thread::hardware_concurrency(),
      int                          max_retries = 3);

    /** @brief Train a forest and return it as a Model::Ptr. */
    static Model::Ptr make(
      const TrainingSpec&          training_spec,
      const types::FeatureMatrix&  x,
      const types::ResponseVector& y,
      int                          size,
      int                          seed,
      int                          n_threads   = std::thread::hardware_concurrency(),
      int                          max_retries = 3);

    std::vector<BootstrapTree::Ptr> trees;
    TrainingSpec::Ptr training_spec;
    const int seed = 0;

    Forest();
    Forest(TrainingSpec::Ptr&& training_spec, int seed);

    /**
     * @brief Predict a single observation.
     *
     * @param data  Feature vector (p).
     * @return      Prediction.
     */
    types::Response predict(const types::FeatureVector& data) const override;

    /**
     * @brief Predict a matrix of observations.
     *
     * @param data  Feature matrix (n × p).
     * @return      Predictions (n).
     */
    types::ResponseVector predict(const types::FeatureMatrix& data) const override;

    /**
     * @brief Predict vote proportions for a matrix of observations.
     *
     * For each observation, counts votes from every tree and returns
     * the proportion of trees that voted for each group.  The number
     * of groups G is derived from the root node of the first tree.
     *
     * @param data  Feature matrix (n × p).
     * @return      Vote proportions matrix (n × G), rows sum to 1.0.
     */
    types::FeatureMatrix predict(const types::FeatureMatrix& data, Proportions) const override;

    /**
     * @brief Add a tree to the forest.
     *
     * @param tree  Tree to add (ownership transferred).
     */
    void add_tree(std::unique_ptr<BootstrapTree> tree);

    bool operator==(const Forest& other) const;
    bool operator!=(const Forest& other) const;

    void accept(Model::Visitor& visitor) const override;

    /**
     * @brief Out-of-bag predictions by majority vote.
     *
     * For each observation, predicts using majority vote of only the
     * trees where it was out-of-bag.  Observations with no OOB tree
     * receive a sentinel value (−1) and are excluded when computing
     * oob_error().
     *
     * @param x  Training feature matrix (n × p).
     * @return   OOB predictions (n); −1 where no OOB tree exists.
     */
    types::ResponseVector oob_predict(const types::FeatureMatrix& x) const;

    /**
     * @brief Out-of-bag error rate.
     *
     * For each observation, predicts using majority vote of only the
     * trees where it was out-of-bag, then computes the overall
     * misclassification rate.
     *
     * @param x  Training feature matrix (n × p).
     * @param y  Training response vector (n).
     * @return   Error rate in [0, 1], or −1 if no observation has any
     *           OOB tree.
     */
    double oob_error(
      const types::FeatureMatrix&  x,
      const types::ResponseVector& y) const;
  };
}
