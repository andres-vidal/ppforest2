/**
 * @file Uniform.test.cpp
 * @brief Statistical correctness tests for the Uniform random integer generator.
 *
 * These tests validate that Uniform (Lemire's nearly-divisionless method)
 * and distinct() (Fisher-Yates shuffle) produce outputs with the correct
 * distributional properties.  Every test uses a fixed seed, making results
 * fully deterministic and reproducible across platforms.
 *
 * The test suite is organised into five groups:
 *
 *   1. **Moment tests** — verify that sample mean and variance converge to
 *      their theoretical values for large N, catching systematic bias in
 *      the generation algorithm.
 *
 *   2. **Chi-squared goodness-of-fit** — the standard frequentist test for
 *      discrete distributions.  We bin observed counts and compare against
 *      the expected uniform frequency via Pearson's chi-squared statistic.
 *      The null hypothesis (correct uniformity) is rejected when
 *      χ² > χ²_crit(df, α).  We use α = 0.001 so that a correct
 *      implementation essentially never produces a false rejection under
 *      a fixed seed.
 *
 *   3. **Range checks** — deterministic boundary tests ensuring every
 *      output falls within [min, max].  These catch off-by-one errors in
 *      the Lemire mapping (e.g. gen_lemire returning s instead of s−1).
 *
 *   4. **distinct() property checks** — verify the sampling-without-
 *      replacement contract: correct count, no duplicates, all values
 *      within range, and full coverage when count equals the range size.
 *
 *   5. **Permutation uniformity** — for a small range [0, K−1], every one
 *      of the K! permutations produced by distinct(K) should appear with
 *      equal probability.  We test this with a chi-squared test over all
 *      K! bins.  This catches subtle Fisher-Yates bugs such as using the
 *      wrong swap range (e.g. swap(i, rand(0, N−1)) instead of
 *      swap(i, rand(0, i))).
 *
 * ## Why fixed seeds make these tests deterministic
 *
 * Statistical tests are inherently probabilistic — a correct generator
 * can still produce a sample that exceeds the critical value.  By fixing
 * the seed, the sample is the same on every run, so the chi-squared
 * statistic is a constant.  If it passes once, it passes always.  If the
 * implementation changes and the constant crosses the critical threshold,
 * the test fails, signalling a distributional regression.
 *
 * ## Tolerance derivation for moment tests
 *
 * For N i.i.d. draws from Uniform{a, …, b}, the CLT gives:
 *
 *   SE(mean)     = σ / √N
 *   SE(variance) ≈ σ² √(2/(N−1))        (for large N)
 *
 * where σ² = ((b−a+1)² − 1) / 12.  Tolerances are computed as exactly
 * 5 SE in the test body, wide enough that a correct implementation
 * under any fixed seed will pass, yet tight enough to catch a biased
 * generator.
 *
 * @see Lemire, "Fast Random Integer Generation in an Interval"
 *      (ACM TOMS, 2019). https://arxiv.org/abs/1805.10941
 * @see Knuth, "The Art of Computer Programming", Vol. 2, §3.4.2
 *      (Fisher-Yates shuffle).
 */

#include <gtest/gtest.h>

#include <cmath>
#include <map>
#include <numeric>
#include <set>
#include <vector>

#include "stats/Uniform.hpp"

using namespace ppforest2::stats;

// ---------------------------------------------------------------------------
// Moment tests
//
// The first two moments of a discrete Uniform{a, …, b} are:
//   E[X]   = (a + b) / 2
//   Var[X] = ((b − a + 1)² − 1) / 12
//
// By the law of large numbers the sample moments converge to these values.
// We draw N = 100 000 samples and assert convergence within a ~5 SE band.
// ---------------------------------------------------------------------------

TEST(Uniform, MeanConvergesToTheory) {
  RNG rng(0);
  Uniform uniform(0, 9);

  constexpr int N = 100000;
  double sum      = 0;

  for (int i = 0; i < N; i++) {
    sum += uniform(rng);
  }

  double sample_mean   = sum / N;
  double expected_mean = 4.5;                 // (0 + 9) / 2
  double se            = std::sqrt(8.25 / N); // σ² = (10² − 1)/12 = 8.25,  SE(mean) = √(σ²/N)
  ASSERT_NEAR(sample_mean, expected_mean, 5 * se);
}

TEST(Uniform, VarianceConvergesToTheory) {
  RNG rng(0);
  Uniform uniform(0, 9);

  constexpr int N = 100000;
  std::vector<int> samples(N);

  for (int i = 0; i < N; i++) {
    samples[i] = uniform(rng);
  }

  double mean   = std::accumulate(samples.begin(), samples.end(), 0.0) / N;
  double sum_sq = 0;

  for (int x : samples) {
    sum_sq += (x - mean) * (x - mean);
  }

  double sample_var   = sum_sq / (N - 1);
  double expected_var = 8.25;                                    // ((b−a+1)² − 1) / 12  =  (100 − 1) / 12
  double se           = expected_var * std::sqrt(2.0 / (N - 1)); // SE(s²) ≈ σ² √(2/(N−1))
  ASSERT_NEAR(sample_var, expected_var, 5 * se);
}

// ---------------------------------------------------------------------------
// Chi-squared goodness-of-fit
//
// Pearson's chi-squared statistic for K equal-probability bins:
//
//   χ² = Σ_i (O_i − E)² / E      where  E = N / K
//
// Under H₀ (true uniformity), χ² ~ χ²(K − 1).  We reject H₀ when
// χ² > χ²_crit(df, α).  Using α = 0.001 gives a very conservative
// threshold — a correct generator passes with probability 0.999 per
// random seed, and with probability 1.0 under a fixed seed once verified.
//
// Critical values (from standard chi-squared tables):
//   df = 2:   χ²_crit(0.001) = 13.816
//   df = 9:   χ²_crit(0.001) = 27.877
// ---------------------------------------------------------------------------

TEST(Uniform, ChiSquaredGoodnessOfFit) {
  RNG rng(0);
  Uniform uniform(0, 9);

  constexpr int N = 100000;
  constexpr int K = 10;
  std::vector<int> counts(K, 0);

  for (int i = 0; i < N; i++) {
    int val = uniform(rng);
    ASSERT_GE(val, 0);
    ASSERT_LE(val, 9);
    counts[val]++;
  }

  double expected = static_cast<double>(N) / K;
  double chi2     = 0;

  for (int i = 0; i < K; i++) {
    double diff = counts[i] - expected;
    chi2 += diff * diff / expected;
  }

  // df = K−1 = 9,  α = 0.001  →  χ²_crit = 27.877
  ASSERT_LT(chi2, 27.877);
}

/**
 * Small-range variant: Uniform{0, 1, 2} with only 3 outcomes.
 *
 * Small ranges stress the rejection branch of Lemire's method because the
 * range size does not evenly divide 2³², making the modular bias correction
 * more likely to activate.  N = 90 000 gives exactly 30 000 expected per bin.
 */
TEST(Uniform, ChiSquaredSmallRange) {
  RNG rng(7);
  Uniform uniform(0, 2);

  constexpr int N = 90000;
  constexpr int K = 3;
  std::vector<int> counts(K, 0);

  for (int i = 0; i < N; i++) {
    counts[uniform(rng)]++;
  }

  double expected = static_cast<double>(N) / K;
  double chi2     = 0;

  for (int i = 0; i < K; i++) {
    double diff = counts[i] - expected;
    chi2 += diff * diff / expected;
  }

  // df = 2,  α = 0.001  →  χ²_crit = 13.816
  ASSERT_LT(chi2, 13.816);
}

// ---------------------------------------------------------------------------
// Range checks
//
// Pure boundary tests: every generated value must satisfy min ≤ val ≤ max.
// These catch off-by-one errors in the Lemire mapping (the result is
// ⌊x·s / 2³²⌋ which must lie in [0, s−1]) and in the min-offset shift.
// ---------------------------------------------------------------------------

TEST(Uniform, AllValuesInRange) {
  RNG rng(123);
  Uniform uniform(5, 15);

  for (int i = 0; i < 10000; i++) {
    int val = uniform(rng);
    ASSERT_GE(val, 5);
    ASSERT_LE(val, 15);
  }
}

/**
 * Degenerate case: when min == max, every draw must return that value.
 * Lemire's method is called with s = 1, which must always yield 0.
 */
TEST(Uniform, SingleValueRange) {
  RNG rng(0);
  Uniform uniform(7, 7);

  for (int i = 0; i < 100; i++) {
    ASSERT_EQ(uniform(rng), 7);
  }
}

// ---------------------------------------------------------------------------
// Batch generation
//
// The multi-sample overload operator()(count, rng) should produce the same
// distributional properties as repeated single draws.
// ---------------------------------------------------------------------------

TEST(Uniform, BatchMeanConverges) {
  RNG rng(0);
  Uniform uniform(0, 99);

  auto samples = uniform(100000, rng);

  double sum         = std::accumulate(samples.begin(), samples.end(), 0.0);
  double sample_mean = sum / samples.size();

  // E[X] = (0 + 99) / 2 = 49.5,  σ² = (100² − 1) / 12 = 833.25
  double se = std::sqrt(833.25 / 100000);
  ASSERT_NEAR(sample_mean, 49.5, 5 * se);
}

// ---------------------------------------------------------------------------
// distinct() — property checks
//
// distinct(count, rng) performs sampling without replacement via
// Fisher-Yates (Knuth) shuffle.  The contract guarantees:
//   (a) exactly `count` values returned,
//   (b) all values in [min, max],
//   (c) no duplicates,
//   (d) when count == range size, the result is a permutation of the
//       full range.
// ---------------------------------------------------------------------------

TEST(UniformDistinct, NoDuplicates) {
  RNG rng(0);
  Uniform uniform(0, 99);

  for (int trial = 0; trial < 100; trial++) {
    auto vals = uniform.distinct(20, rng);
    std::set<int> unique_vals(vals.begin(), vals.end());
    ASSERT_EQ(unique_vals.size(), 20u);
  }
}

TEST(UniformDistinct, AllInRange) {
  RNG rng(0);
  Uniform uniform(10, 50);

  for (int trial = 0; trial < 100; trial++) {
    auto vals = uniform.distinct(15, rng);

    for (int v : vals) {
      ASSERT_GE(v, 10);
      ASSERT_LE(v, 50);
    }
  }
}

TEST(UniformDistinct, CorrectCount) {
  RNG rng(0);
  Uniform uniform(0, 19);

  ASSERT_EQ(uniform.distinct(10, rng).size(), 10u);
  ASSERT_EQ(uniform.distinct(20, rng).size(), 20u);
  ASSERT_EQ(uniform.distinct(1, rng).size(), 1u);
  ASSERT_EQ(uniform.distinct(0, rng).size(), 0u);
}

TEST(UniformDistinct, FullRangeReturnsAllValues) {
  RNG rng(0);
  Uniform uniform(0, 9);

  auto vals = uniform.distinct(10, rng);
  std::set<int> unique_vals(vals.begin(), vals.end());

  ASSERT_EQ(unique_vals.size(), 10u);

  for (int i = 0; i < 10; i++) {
    ASSERT_TRUE(unique_vals.count(i));
  }
}

// ---------------------------------------------------------------------------
// distinct() — permutation uniformity (chi-squared)
//
// For a range of size K, distinct(K) produces one of K! possible
// permutations.  If the shuffle is correct, each permutation has
// probability exactly 1/K!.
//
// We draw N permutations, count how often each appears, and apply the
// chi-squared test with K!−1 degrees of freedom.  This is the definitive
// test for Fisher-Yates correctness — a common bug is using the wrong
// swap range (e.g. swap(i, rand(0, N−1)) instead of swap(i, rand(0, i))),
// which produces a non-uniform distribution over permutations even though
// every element appears with equal marginal frequency.
//
// We use K = 4 (24 permutations).  N = 24 000 gives an expected count
// of 1 000 per permutation, well above the χ² rule-of-thumb minimum of 5.
//
// Critical value:  df = 23,  α = 0.001  →  χ²_crit = 49.728
// ---------------------------------------------------------------------------

TEST(UniformDistinct, PermutationUniformity) {
  RNG rng(0);
  Uniform uniform(0, 3);

  // 4! = 24 possible permutations
  constexpr int N_PERMS = 24;
  constexpr int N       = 24000;

  std::map<std::vector<int>, int> perm_counts;

  for (int i = 0; i < N; i++) {
    auto perm = uniform.distinct(4, rng);
    perm_counts[perm]++;
  }

  // every permutation must appear
  ASSERT_EQ(perm_counts.size(), static_cast<size_t>(N_PERMS));

  double expected = static_cast<double>(N) / N_PERMS;
  double chi2     = 0;

  for (auto& [perm, count] : perm_counts) {
    double diff = count - expected;
    chi2 += diff * diff / expected;
  }

  // df = 23,  α = 0.001  →  χ²_crit = 49.728
  ASSERT_LT(chi2, 49.728);
}
