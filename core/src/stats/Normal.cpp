#include "stats/Normal.hpp"

#include <cmath>

using ppforest2::types::Feature;

namespace ppforest2::stats {
  double Normal::gen_unif01(RNG& rng) {
    static constexpr uint64_t PRECISION = 53;
    static constexpr uint64_t MAX_BITS  = (1ULL << PRECISION) - 1;

    uint64_t bits = 0;

    while (bits == 0 || bits == MAX_BITS) {
      uint64_t x = rng();
      uint64_t y = rng();
      bits       = ((x << 21) | (y >> 11)) & MAX_BITS;
    }

    return static_cast<double>(bits) / static_cast<double>(MAX_BITS + 1);
  }

  Feature Normal::denormalize(Feature z) const {
    return mean + std_dev * z;
  }

  Normal::Normal(Feature mean, Feature std_dev)
      : mean(mean)
      , std_dev(std_dev) {}

  Feature Normal::operator()(RNG& rng) {
    if (cached_z.has_value()) {
      Feature z = cached_z.value();
      cached_z.reset();
      return denormalize(z);
    }

    double u1 = gen_unif01(rng);
    double u2 = gen_unif01(rng);

    double r     = std::sqrt(-2.0 * std::log(u1));
    double theta = 2.0 * M_PI * u2;

    double z1 = r * std::cos(theta);
    double z2 = r * std::sin(theta);

    cached_z = static_cast<Feature>(z2);

    return denormalize(static_cast<float>(z1));
  }

  std::vector<Feature> Normal::operator()(int count, RNG& rng) {
    std::vector<Feature> result(count);

    for (int i = 0; i < count; i++) {
      result[i] = operator()(rng);
    }

    return result;
  }
}
