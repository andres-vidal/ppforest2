#pragma once

#include "stats/Stats.hpp"
#include "stats/DataPacket.hpp"
#include "utils/Types.hpp"

#include <vector>

namespace ppforest2::stats {
  namespace simulation::params {
    /**
     * @brief Classification simulation: group-shifted normals.
     *
     * Each row's features are drawn from `Normal(mean + g * mean_separation, sd)`
     * where `g = i % G`. The response is the group label.
     */
    struct Classification {
      types::Feature mean            = 100.0F; ///< Base mean for the first group.
      types::Feature mean_separation = 50.0F;  ///< Mean shift between successive groups.
      types::Feature sd              = 10.0F;  ///< Standard deviation within each group.
    };

    /**
     * @brief Regression simulation: linear model over i.i.d. features.
     *
     * Features drawn from `Normal(0, sd)`; response is
     * `y_intercept + Σ coef_j * x_j + noise` for the first `n_informative`
     * features (default `min(p, 5)`). Coefficients are deterministic
     * `coef_j = j + 1` for reproducibility.
     */
    struct Regression {
      int n_informative          = 0;    ///< Informative feature count (0 → min(p, 5)).
      types::Feature y_intercept = 0.0F; ///< Base intercept added to every response.
      types::Feature y_sd        = 0.1F; ///< Standard deviation of response noise.
      types::Feature sd          = 1.0F; ///< Standard deviation of feature values.
    };
  }

  /**
   * @brief Generate a simulated classification dataset.
   *
   * Rows are sorted by group label.
   */
  DataPacket simulate(int n, int p, int G, RNG& rng, simulation::params::Classification const& params = {});

  /**
   * @brief Generate a simulated regression dataset.
   *
   * Rows are sorted by the continuous response. Distinguished from the
   * classification overload by the absence of a group-count parameter.
   */
  DataPacket simulate(int n, int p, RNG& rng, simulation::params::Regression const& params = {});

  /**
   * @brief Indices for a train/test split.
   */
  struct Split {
    std::vector<int> tr; ///< Training set indices.
    std::vector<int> te; ///< Test set indices.
  };

  /**
   * @brief Perform a stratified random train/test split on a DataPacket.
   *
   * Samples indices within each group proportional to train_ratio so that
   * group balance is preserved in both train and test sets.
   *
   * @param data        The full dataset.
   * @param train_ratio Proportion of data to use for training (0, 1).
   * @param rng         Random number generator.
   * @return A Split containing train and test index vectors.
   */
  Split split(DataPacket const& data, float train_ratio, RNG& rng);
}
