#include "stats/Simulation.hpp"
#include "stats/Normal.hpp"
#include "stats/Uniform.hpp"
#include "stats/GroupPartition.hpp"

#include <algorithm>
#include <string>

using namespace ppforest2::types;
using namespace ppforest2::stats::simulation;

namespace ppforest2::stats {
  namespace {
    // Generate an n × p feature matrix where row i is drawn from
    // Normal(base_mean + (i % G) * separation, sd). With G = 1 and
    // separation = 0 this degenerates to i.i.d. Normal(base_mean, sd),
    // which is what the regression overload wants.
    FeatureMatrix generate_features(int n, int p, int G, Feature mean, Feature sep, Feature sd, RNG& rng) {
      FeatureMatrix x(n, p);
      for (int i = 0; i < n; ++i) {
        Normal norm(mean + (static_cast<Feature>(i % G)) * sep, sd);
        for (int j = 0; j < p; ++j) {
          x(i, j) = norm(rng);
        }
      }
      return x;
    }
  }

  DataPacket simulate(int const n, int const p, int const G, RNG& rng, params::Classification const& params) {
    auto x = generate_features(n, p, G, params.mean, params.mean_separation, params.sd, rng);

    GroupIdVector y_int(n);
    for (int i = 0; i < n; ++i) {
      y_int[i] = i % G;
    }

    sort(x, y_int);

    OutcomeVector y = y_int.cast<Outcome>();

    // Synthesize placeholder group names so the returned DataPacket upholds
    // the classification invariant "group_names has one entry per class."
    // Without this, `group_names` would be `[]`, which cascades into
    // out-of-bounds reads in labeled serialization paths (see
    // `Export<Model::Ptr>::to_json`). Real-data readers (e.g. CSV) populate
    // names from the label column — simulation has no source, so we fall
    // back to the integer index as a string.
    std::vector<std::string> group_names;
    group_names.reserve(static_cast<std::size_t>(G));
    for (int g = 0; g < G; ++g) {
      group_names.emplace_back(std::to_string(g));
    }

    return DataPacket(x, y, /*group_names=*/group_names, /*feature_names=*/{});
  }

  Split split(DataPacket const& data, float train_ratio, RNG& rng) {
    auto const n          = data.x.rows();
    auto const train_size = static_cast<int>(static_cast<float>(n) * train_ratio);

    std::vector<int> train_indices;
    std::vector<int> test_indices;

    train_indices.reserve(train_size);
    test_indices.reserve(n - train_size);

    if (data.groups.empty()) {
      // Regression path: `data.groups` is empty by construction (no
      // discrete class labels to stratify on). Draw a uniform random
      // subset of row indices, then sort each side so that callers
      // which rely on y-sorted contiguity (e.g. `ByCutpoint::init`
      // at the root of the regression training pipeline) still see
      // the correct shape.
      Uniform unif(0, n - 1);
      std::vector<int> indices = unif.distinct(n, rng);

      std::vector<int> tr(indices.begin(), indices.begin() + train_size);
      std::vector<int> te(indices.begin() + train_size, indices.end());

      // Preserve the y-sorted row ordering on each side. `data.y` is
      // already sorted ascending (regression DataPackets are produced
      // that way by `io::csv::read_regression_sorted` and
      // `simulate_regression`), so sorting by index is equivalent to
      // sorting by y.
      std::sort(tr.begin(), tr.end());
      std::sort(te.begin(), te.end());

      return {.tr = std::move(tr), .te = std::move(te)};
    }

    // Classification path: stratified per-group split to preserve class
    // balance in both sides.
    GroupPartition spec(data.y);

    for (auto const& group : data.groups) {
      int group_start      = spec.group_start(group);
      int group_size       = spec.group_size(group);
      int group_end        = group_start + group_size - 1;
      int group_train_size = static_cast<int>(static_cast<float>(group_size) * train_ratio);

      Uniform unif(group_start, group_end);
      std::vector<int> group_indices = unif.distinct(group_size, rng);

      train_indices.insert(train_indices.end(), group_indices.begin(), group_indices.begin() + group_train_size);
      test_indices.insert(test_indices.end(), group_indices.begin() + group_train_size, group_indices.end());
    }

    return {.tr = train_indices, .te = test_indices};
  }

  DataPacket simulate(int const n, int const p, RNG& rng, params::Regression const& params) {
    using namespace types;

    // Regression = "classification with G=1": all rows drawn from one
    // distribution, with a response overlaid on top.
    auto x = generate_features(n, p, 1, Feature(0), Feature(0), params.sd, rng);

    int const n_informative = params.n_informative > 0 ? std::min(params.n_informative, p) : std::min(p, 5);

    // Deterministic coefficients for the first n_informative features —
    // reproducible across runs with the same n_informative.
    std::vector<Feature> coef(static_cast<std::size_t>(n_informative));
    for (int j = 0; j < n_informative; ++j) {
      coef[static_cast<std::size_t>(j)] = static_cast<Feature>(j + 1);
    }

    // Linear response + noise.
    Normal noise(Feature(0), params.y_sd);
    OutcomeVector y(n);
    for (int i = 0; i < n; ++i) {
      Feature yi = params.y_intercept;
      for (int j = 0; j < n_informative; ++j) {
        yi += coef[static_cast<std::size_t>(j)] * x(i, j);
      }
      y(i) = yi + noise(rng);
    }

    sort(x, y);
    return DataPacket(x, y, std::set<GroupId>{});
  }
}
