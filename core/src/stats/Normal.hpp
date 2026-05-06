#pragma once

#include <optional>
#include <vector>

#include "stats/Stats.hpp"
#include "utils/Types.hpp"

namespace ppforest2::stats {
  /**
   * @brief Normal (Gaussian) random number generator.
   *
   * Generates samples from N(mean, std_dev²) using the Box-Muller
   * transform.  Each call to the RNG produces a pair of independent
   * standard normal variates; the second is cached and returned on
   * the next invocation, so roughly half of the calls avoid the
   * expensive trigonometric computation.
   *
   * Uniform inputs are generated internally at 53-bit precision
   * (see gen_unif01()), matching the significand width of IEEE 754
   * double-precision floats.
   */
  class Normal {
  private:
    types::Feature mean;
    types::Feature std_dev;

    std::optional<types::Feature> cached_z;

    /**
       * @brief Generate a uniform random number in (0, 1).
       *
       * Combines two 32-bit pcg32 outputs into a 53-bit integer and
       * normalizes to the unit interval.  Rejects the boundary values
       * 0 and 1 so that log(U) and log(1−U) are always finite.
       *
       * @param rng  Random number generator (pcg32).
       * @return     Uniform variate in (0, 1).
       */
    double gen_unif01(RNG& rng);

    /**
       * @brief Transform a standard normal variate to N(mean, std_dev²).
       *
       * @param z  Standard normal variate.
       * @return   mean + std_dev * z.
       */
    types::Feature denormalize(types::Feature z) const;

  public:
    /**
       * @brief Construct a Normal generator.
       *
       * @param mean     Distribution mean.
       * @param std_dev  Distribution standard deviation.
       */
    Normal(types::Feature mean, types::Feature std_dev);

    /**
       * @brief Generate a single normal variate via the Box-Muller transform.
       *
       * Draws two independent uniform variates U1, U2 ∈ (0, 1) and
       * computes a pair of standard normal variates:
       *
       *   Z1 = √(−2 ln U1) · cos(2π U2)
       *   Z2 = √(−2 ln U1) · sin(2π U2)
       *
       * Returns Z1 (after denormalization) and caches Z2 for the next
       * call.  When a cached value is available it is returned directly
       * without generating new uniform inputs.
       *
       * @param rng  Random number generator (pcg32).
       * @return     A sample from N(mean, std_dev²).
       */
    types::Feature operator()(RNG& rng);

    /**
       * @brief Generate multiple normal variates.
       *
       * @param count  Number of samples to generate.
       * @param rng    Random number generator (pcg32).
       * @return       Vector of @p count samples from N(mean, std_dev²).
       */
    std::vector<types::Feature> operator()(int count, RNG& rng);
  };
}
