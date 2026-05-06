#include "stats/Stats.hpp"
#include "utils/Invariant.hpp"

#include <Eigen/Dense>
#include <map>

using namespace ppforest2::types;

namespace ppforest2::stats {
  std::set<GroupId> unique(GroupIdVector const& column) {
    std::set<GroupId> unique_values;

    for (int i = 0; i < column.rows(); i++) {
      unique_values.insert(column(i));
    }

    return unique_values;
  }

  FeatureVector var(FeatureMatrix const& data) {
    invariant(data.rows() >= 2, "var: matrix must have at least 2 rows");

    FeatureMatrix centered = data.rowwise() - data.colwise().mean();
    return centered.array().square().colwise().sum() / static_cast<Feature>(data.rows() - 1);
  }

  FeatureVector sd(FeatureMatrix const& data) {
    return var(data).array().sqrt();
  }

  Outcome majority_vote(std::vector<Outcome> const& preds) {
    invariant(!preds.empty(), "majority_vote: preds must be non-empty");

    std::map<GroupId, int> votes;
    for (auto p : preds) {
      votes[static_cast<GroupId>(p)] += 1;
    }

    GroupId best   = 0;
    int best_count = 0;
    for (auto const& [cls, cnt] : votes) {
      if (cnt > best_count) {
        best       = cls;
        best_count = cnt;
      }
    }
    return static_cast<Outcome>(best);
  }

  Outcome mean(std::vector<Outcome> const& preds) {
    invariant(!preds.empty(), "mean: preds must be non-empty");

    double sum = 0.0;
    for (auto p : preds) {
      sum += static_cast<double>(p);
    }
    return static_cast<Outcome>(sum / static_cast<double>(preds.size()));
  }

  std::map<GroupId, int> group_indices(std::set<GroupId> const& groups) {
    std::map<GroupId, int> indices;
    int g = 0;
    for (auto const& key : groups) {
      indices[key] = g++;
    }
    return indices;
  }
}
