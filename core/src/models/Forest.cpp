#include "models/Forest.hpp"
#include "stats/Stats.hpp"
#include "utils/Types.hpp"
#include "utils/Invariant.hpp"

#include <algorithm>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace ppforest2::stats;
using namespace ppforest2::types;

namespace ppforest2 {
  Forest Forest::train(
    const TrainingSpec &  training_spec,
    const FeatureMatrix & x,
    const ResponseVector& y,
    const int             size,
    const int             seed,
    const int             n_threads,
    const int             max_retries) {
    #ifdef _OPENMP
    omp_set_num_threads(n_threads);
    #endif

    GroupPartition group_spec(y);

    invariant(size > 0, "The forest size must be greater than 0.");

    std::vector<BootstrapTree::Ptr> trees(size);
    std::vector<std::exception_ptr> errors(size);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < size; i++) {
      for (int attempt = 0; attempt <= max_retries; ++attempt) {
        try {
          uint64_t stream = static_cast<uint64_t>(i) + static_cast<uint64_t>(attempt) * static_cast<uint64_t>(size);
          RNG rng(static_cast<uint64_t>(seed), stream);
          trees[i] = BootstrapTree::train(training_spec, x, group_spec, rng);

          if (!trees[i]->degenerate) {
            break;
          }
        } catch (...) {
          errors[i] = std::current_exception();
          break;
        }
      }
    }

    Forest forest(training_spec.clone(), seed);

    for (int i = 0; i < size; i++) {
      if (errors[i]) {
        std::rethrow_exception(errors[i]);
      }

      if (trees[i]->degenerate) {
        forest.degenerate = true;
      }

      forest.add_tree(std::move(trees[i]));
    }

    return forest;
  }

  Forest::Forest() {
  }

  Forest::Forest(TrainingSpec::Ptr&& training_spec, int seed)
    : training_spec(std::move(training_spec)),
    seed(seed) {
  }

  Response Forest::predict(const FeatureVector& data) const {
    std::map<Response, int> votes_per_group;

    for (const auto& tree : trees) {
      Response prediction = tree->predict(data);
      votes_per_group[prediction] += 1;
    }

    Response best  = 0;
    int best_count = 0;

    for (const auto& [key, votes] : votes_per_group) {
      if (votes > best_count) {
        best       = key;
        best_count = votes;
      }
    }

    return best;
  }

  ResponseVector Forest::predict(const FeatureMatrix& data) const {
    ResponseVector predictions(data.rows());

    for (int i = 0; i < data.rows(); i++) {
      predictions(i) = predict((FeatureVector)data.row(i));
    }

    return predictions;
  }

  FeatureMatrix Forest::predict(const FeatureMatrix& data, Proportions) const {
    invariant(!trees.empty(), "Forest has no trees.");

    std::set<Response> group_set = trees[0]->root->node_groups();
    std::vector<Response> groups(group_set.begin(), group_set.end());
    int G = static_cast<int>(groups.size());

    std::map<Response, int> group_to_col;
    for (int g = 0; g < G; ++g) {
      group_to_col[groups[static_cast<std::size_t>(g)]] = g;
    }

    int n                     = static_cast<int>(data.rows());
    FeatureMatrix proportions = FeatureMatrix::Zero(n, G);

    for (int i = 0; i < n; ++i) {
      for (const auto& tree : trees) {
        Response pred = tree->predict((FeatureVector)data.row(i));
        auto it       = group_to_col.find(pred);

        if (it != group_to_col.end()) {
          proportions(i, it->second) += 1;
        }
      }

      Feature total = proportions.row(i).sum();

      if (total > 0) {
        proportions.row(i) /= total;
      }
    }

    return proportions;
  }

  void Forest::add_tree(std::unique_ptr<BootstrapTree> tree) {
    trees.push_back(std::move(tree));
  }

  bool Forest::operator==(const Forest& other) const {
    if (trees.size() != other.trees.size()) {
      return false;
    }

    for (std::size_t i = 0; i < trees.size(); i++) {
      if (*trees[i] != *other.trees[i]) {
        return false;
      }
    }

    return true;
  }

  bool Forest::operator!=(const Forest& other) const {
    return !(*this == other);
  }

  void Forest::accept(Model::Visitor& visitor) const {
    visitor.visit(*this);
  }

  ResponseVector Forest::oob_predict(const FeatureMatrix& x) const {
    const int n_total = static_cast<int>(x.rows());
    const int B       = static_cast<int>(trees.size());

    std::vector<std::map<Response, int>> votes(static_cast<std::size_t>(n_total));

    for (int k = 0; k < B; ++k) {
      const BootstrapTree& tree    = *trees[k];
      std::vector<int>     oob_idx = tree.oob_indices(n_total);
      ResponseVector preds         = tree.predict_oob(x, oob_idx);

      for (int j = 0; j < static_cast<int>(oob_idx.size()); ++j) {
        int i = oob_idx[static_cast<std::size_t>(j)];
        votes[static_cast<std::size_t>(i)][preds(j)] += 1;
      }
    }

    ResponseVector out(n_total);
    out.fill(-1);

    for (int i = 0; i < n_total; ++i) {
      const auto& obs_votes = votes[static_cast<std::size_t>(i)];

      if (obs_votes.empty()) {
        continue;
      }

      Response best  = 0;
      int best_count = 0;

      for (const auto& [cls, cnt] : obs_votes) {
        if (cnt > best_count) {
          best       = cls;
          best_count = cnt;
        }
      }

      out(i) = best;
    }

    return out;
  }

  double Forest::oob_error(
    const FeatureMatrix&  x,
    const ResponseVector& y) const {
    ResponseVector preds = oob_predict(x);

    std::vector<int> oob_rows;

    for (int i = 0; i < preds.size(); ++i) {
      if (preds(i) >= 0) {
        oob_rows.push_back(i);
      }
    }

    if (oob_rows.empty()) {
      return -1.0;
    }

    ResponseVector preds_oob = preds(oob_rows, Eigen::all).eval();
    ResponseVector y_oob     = y(oob_rows, Eigen::all).eval();

    return stats::error_rate(preds_oob, y_oob);
  }
}
