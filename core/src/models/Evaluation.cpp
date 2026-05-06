#include "models/Evaluation.hpp"

#include "models/Bagged.hpp"
#include "models/ClassificationForest.hpp"
#include "models/ClassificationTree.hpp"
#include "models/Forest.hpp"
#include "models/Model.hpp"
#include "models/RegressionForest.hpp"
#include "models/RegressionTree.hpp"
#include "models/Tree.hpp"
#include "models/VIVisitor.hpp"
#include "stats/Metrics.hpp"
#include "stats/Stats.hpp"
#include "stats/Uniform.hpp"
#include "utils/UserError.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <limits>
#include <map>
#include <set>

using namespace ppforest2::types;
using namespace ppforest2::stats;

namespace ppforest2 {
  namespace {
    // -------------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------------

    // Population variance (n divisor, matches stats::mse so that an
    // "always-predict-mean" baseline gives NMSE = 1).
    double population_variance(OutcomeVector const& v) {
      if (v.size() == 0) {
        return 0.0;
      }
      double const mean = static_cast<double>(v.mean());
      return (v.cast<double>().array() - mean).square().mean();
    }

    // -------------------------------------------------------------------------
    // OOB primitive
    //
    // `OOBPredictions` carries only the rows that had at least one OOB tree
    // voting on them, paired with the aggregated prediction for each. Every
    // higher-level OOB function (`oob_predict`, `oob_error`, `oob_metrics`)
    // is built on top — the sentinel-padded `oob_predict` form is just this
    // primitive scattered back into a full-length vector at the boundary.
    // No callers need to know what the "no OOB tree" sentinel is, or even
    // that there is one.
    // -------------------------------------------------------------------------

    struct OOBPredictions {
      std::vector<int> rows;     // original-row indices with ≥ 1 OOB tree
      OutcomeVector predictions; // aggregated prediction per `rows[i]`
    };

    // Shared OOB-aggregation skeleton. Collects per-tree OOB predictions per
    // row, then applies @p aggregate to the rows that received at least one
    // contribution. Rows with zero contributions are dropped from the output.
    template<typename Aggregate>
    OOBPredictions oob_predictions_with(Forest const& forest, FeatureMatrix const& x, Aggregate aggregate) {
      int const n_total = static_cast<int>(x.rows());

      std::vector<std::vector<Outcome>> per_row(static_cast<std::size_t>(n_total));

      for (auto const& bag_ptr : forest.trees) {
        BaggedTree const& bag    = *bag_ptr;
        std::vector<int> oob_idx = bag.oob_indices(n_total);
        OutcomeVector preds      = bag.predict_oob(x, oob_idx);

        for (int j = 0; j < static_cast<int>(oob_idx.size()); ++j) {
          per_row[static_cast<std::size_t>(oob_idx[static_cast<std::size_t>(j)])].push_back(preds(j));
        }
      }

      OOBPredictions out;
      out.rows.reserve(static_cast<std::size_t>(n_total));
      std::vector<Outcome> picked;
      picked.reserve(static_cast<std::size_t>(n_total));

      for (int i = 0; i < n_total; ++i) {
        auto const& row_preds = per_row[static_cast<std::size_t>(i)];
        if (row_preds.empty()) {
          continue;
        }
        out.rows.push_back(i);
        picked.push_back(aggregate(row_preds));
      }

      out.predictions = Eigen::Map<OutcomeVector const>(picked.data(), static_cast<Eigen::Index>(picked.size())).eval();
      return out;
    }

    // VI1 (permutation) protocol: a `baseline` callable computes a per-tree
    // context (or returns `nullopt` to skip the tree); a `delta` callable
    // turns each shuffled-column prediction set into the per-variable
    // contribution added to `importance(j)`. Pair the two functions by
    // convention — the classification pair uses error-rate delta, the
    // regression pair uses normalised MSE delta and skips trees whose OOB
    // response is constant.

    std::optional<double> accuracy_baseline(
        BaggedTree const& tree, FeatureMatrix const& x, OutcomeVector const& y_oob, std::vector<int> const& oob_idx
    ) {
      return stats::error_rate(tree.predict_oob(x, oob_idx), y_oob);
    }

    Feature accuracy_delta(double baseline_err, OutcomeVector const& perm_pred, OutcomeVector const& y_oob) {
      return static_cast<Feature>(stats::error_rate(perm_pred, y_oob) - baseline_err);
    }

    struct NMSEContext {
      double baseline_mse;
      double var_y;
    };

    std::optional<NMSEContext> nmse_baseline(
        BaggedTree const& tree, FeatureMatrix const& x, OutcomeVector const& y_oob, std::vector<int> const& oob_idx
    ) {
      double const var_y = population_variance(y_oob);
      if (var_y <= 0.0) {
        return std::nullopt;
      }
      return NMSEContext{stats::mse(tree.predict_oob(x, oob_idx), y_oob), var_y};
    }

    Feature nmse_delta(NMSEContext const& ctx, OutcomeVector const& perm_pred, OutcomeVector const& y_oob) {
      return static_cast<Feature>((stats::mse(perm_pred, y_oob) - ctx.baseline_mse) / ctx.var_y);
    }

    // Shared VI1 (permuted importance) skeleton. For every valid OOB tree:
    // gather y_oob, compute a per-tree baseline context (or skip the tree if
    // baseline returns nullopt), then for each column shuffle in place,
    // predict, score the delta, restore. The platform-stable RNG seed is
    // derived from `seed ^ k` (where `k` is the tree's position).
    template<typename Baseline, typename Delta>
    FeatureVector vi_permuted_with(
        Forest const& forest, FeatureMatrix const& x, OutcomeVector const& y, int seed, Baseline baseline, Delta delta
    ) {
      int const n_vars  = static_cast<int>(x.cols());
      int const n_total = static_cast<int>(x.rows());

      FeatureVector importance = FeatureVector::Zero(n_vars);
      int valid_trees          = 0;

      for_each_bag_with_oob(forest.trees, n_total, [&](BaggedTree const& tree, std::vector<int> const& oob_idx, int k) {
        OutcomeVector const y_oob = y(oob_idx, Eigen::all);
        int const n_oob           = static_cast<int>(oob_idx.size());

        auto ctx_opt = baseline(tree, x, y_oob, oob_idx);
        if (!ctx_opt) {
          return;
        }
        auto const& ctx = *ctx_opt;

        stats::RNG rng(static_cast<unsigned>(seed) ^ static_cast<unsigned>(k));
        stats::Uniform uniform(0, n_oob - 1);

        FeatureMatrix perm_x = x(oob_idx, Eigen::all);
        OutcomeVector perm_pred(n_oob);

        for (int j = 0; j < n_vars; ++j) {
          FeatureVector const col_saved    = perm_x.col(j);
          std::vector<int> const row_order = uniform.distinct(n_oob, rng);

          for (int i = 0; i < n_oob; ++i) {
            perm_x(i, j) = col_saved(row_order[static_cast<std::size_t>(i)]);
          }
          for (int i = 0; i < n_oob; ++i) {
            perm_pred(i) = tree.predict(static_cast<FeatureVector>(perm_x.row(i)));
          }

          importance(j) += delta(ctx, perm_pred, y_oob);

          perm_x.col(j) = col_saved;
        }

        ++valid_trees;
      });

      if (valid_trees > 0) {
        importance /= static_cast<Feature>(valid_trees);
      }
      return importance;
    }

    // Shared VI3 (weighted projections) skeleton. For every valid OOB tree,
    // @p weight_fn returns either a tree weight or `nullopt` to skip; the
    // tree's projection-coefficient contributions (from VIVisitor) are
    // weighted and summed. The final divisor is `valid_trees * group_factor`
    // — classification uses `(G - 1)` to bake in the random-classifier
    // baseline; regression uses `1`.
    template<typename WeightFn>
    FeatureVector vi_weighted_projections_with(
        Forest const& forest,
        FeatureMatrix const& x,
        OutcomeVector const& y,
        FeatureVector const* scale,
        WeightFn weight_fn,
        Feature group_factor
    ) {
      int const n_vars  = static_cast<int>(x.cols());
      int const n_total = static_cast<int>(x.rows());

      FeatureVector importance = FeatureVector::Zero(n_vars);
      int valid_trees          = 0;

      for_each_bag_with_oob(
          forest.trees,
          n_total,
          [&](BaggedTree const& tree, std::vector<int> const& oob_idx, int /*k*/) {
            OutcomeVector const y_oob = y(oob_idx, Eigen::all);

            auto weight_opt = weight_fn(tree, x, y_oob, oob_idx);
            if (!weight_opt) {
              return;
            }
            Feature const weight = *weight_opt;

            VIVisitor visitor(n_vars, scale);
            tree.model->root->accept(visitor);

            for (int j = 0; j < n_vars; ++j) {
              importance(j) += weight * static_cast<Feature>(visitor.vi3_contributions[static_cast<std::size_t>(j)]);
            }

            ++valid_trees;
          }
      );

      Feature const denom = static_cast<Feature>(valid_trees) * group_factor;
      if (denom > Feature(0)) {
        importance /= denom;
      }
      return importance;
    }

    // VI3 weight functions: per-tree score, mapped to a contribution weight.
    // Returning `nullopt` skips the tree (e.g., regression with constant
    // OOB response — the NMSE denominator would be zero).

    std::optional<Feature> accuracy_weight(
        BaggedTree const& tree, FeatureMatrix const& x, OutcomeVector const& y_oob, std::vector<int> const& oob_idx
    ) {
      Feature const err = static_cast<Feature>(stats::error_rate(tree.predict_oob(x, oob_idx), y_oob));
      return Feature(1) - err;
    }

    std::optional<Feature> nmse_clipped_weight(
        BaggedTree const& tree, FeatureMatrix const& x, OutcomeVector const& y_oob, std::vector<int> const& oob_idx
    ) {
      double const var_y = population_variance(y_oob);
      if (var_y <= 0.0) {
        return std::nullopt;
      }
      double const nmse = stats::mse(tree.predict_oob(x, oob_idx), y_oob) / var_y;
      return static_cast<Feature>(std::max(0.0, 1.0 - nmse));
    }

  }

  // ---------------------------------------------------------------------------
  // Public free-function API
  // ---------------------------------------------------------------------------

  FeatureVector vi_projections(Tree const& tree, int n_vars, FeatureVector const* scale) {
    VIVisitor visitor(n_vars, scale);
    tree.root->accept(visitor);
    return Eigen::Map<FeatureVector const>(visitor.vi2_contributions.data(), n_vars);
  }

  FeatureVector vi_projections(Forest const& forest, int n_vars, FeatureVector const* scale) {
    FeatureVector importance = FeatureVector::Zero(n_vars);

    for (auto const& bt : forest.trees) {
      VIVisitor visitor(n_vars, scale);
      bt->model->root->accept(visitor);

      for (int j = 0; j < n_vars; ++j) {
        importance(j) += static_cast<Feature>(visitor.vi2_contributions[static_cast<std::size_t>(j)]);
      }
    }

    if (!forest.trees.empty()) {
      importance /= static_cast<Feature>(forest.trees.size());
    }

    return importance;
  }

  FeatureVector vi_permuted(Forest const& forest, FeatureMatrix const& x, OutcomeVector const& y, int seed) {
    struct Visitor : Model::Visitor {
      FeatureMatrix const& x;
      OutcomeVector const& y;
      int seed;
      FeatureVector result;

      Visitor(FeatureMatrix const& x, OutcomeVector const& y, int seed)
          : x(x)
          , y(y)
          , seed(seed) {}

      void visit(Forest const&) override {
        throw UserError("vi_permuted: unknown forest mode (expected classification or regression).");
      }
      void visit(ClassificationForest const& forest) override {
        result = vi_permuted_with(forest, x, y, seed, accuracy_baseline, accuracy_delta);
      }
      void visit(RegressionForest const& forest) override {
        result = vi_permuted_with(forest, x, y, seed, nmse_baseline, nmse_delta);
      }
    };

    Visitor visitor(x, y, seed);
    forest.accept(visitor);
    return std::move(visitor.result);
  }

  FeatureVector vi_weighted_projections(
      Forest const& forest, FeatureMatrix const& x, OutcomeVector const& y, FeatureVector const* scale
  ) {
    struct Visitor : Model::Visitor {
      FeatureMatrix const& x;
      OutcomeVector const& y;
      FeatureVector const* scale;
      FeatureVector result;

      Visitor(FeatureMatrix const& x, OutcomeVector const& y, FeatureVector const* scale)
          : x(x)
          , y(y)
          , scale(scale) {}

      void visit(Forest const&) override {
        throw UserError("vi_weighted_projections: unknown forest mode (expected classification or regression).");
      }
      void visit(ClassificationForest const& forest) override {
        // The (G - 1) factor in the denominator subtracts the random-classifier
        // baseline implicitly: a classifier predicting at random has error rate
        // (G - 1) / G, so its weight `1 - error_rate` is `1/G`, and the (G - 1)
        // scale lifts informative trees relative to that floor.
        GroupIdVector const y_int  = y.cast<GroupId>();
        Feature const group_factor = static_cast<Feature>(stats::unique(y_int).size() - 1);
        result                     = vi_weighted_projections_with(forest, x, y, scale, accuracy_weight, group_factor);
      }
      void visit(RegressionForest const& forest) override {
        result = vi_weighted_projections_with(forest, x, y, scale, nmse_clipped_weight, Feature(1));
      }
    };

    Visitor visitor(x, y, scale);
    forest.accept(visitor);
    return std::move(visitor.result);
  }

  VariableImportance variable_importance(Tree const& tree, FeatureMatrix const& x) {
    VariableImportance vi;
    vi.scale       = stats::sd(x);
    vi.scale       = (vi.scale.array() > Feature(0)).select(vi.scale, Feature(1));
    vi.projections = vi_projections(tree, static_cast<int>(x.cols()), &vi.scale);
    return vi;
  }

  VariableImportance
  variable_importance(Forest const& forest, FeatureMatrix const& x, OutcomeVector const& y, int seed) {
    VariableImportance vi;
    vi.scale                = stats::sd(x);
    vi.scale                = (vi.scale.array() > Feature(0)).select(vi.scale, Feature(1));
    vi.permuted             = vi_permuted(forest, x, y, seed);
    vi.projections          = vi_projections(forest, static_cast<int>(x.cols()), &vi.scale);
    vi.weighted_projections = vi_weighted_projections(forest, x, y, &vi.scale);
    return vi;
  }

  // Reconstitutes the legacy sentinel-padded OOB prediction vector from the
  // sparse `OOBPredictions` primitive. The sentinel only exists at this
  // boundary — `<no_oob>` rows are filled with `NaN` (in either mode) so
  // the shape matches `x.rows()` for callers that wanted aligned indexing.
  // Prefer `oob_metrics` / `oob_error` if you don't need the alignment —
  // they own the sentinel-filter logic internally.
  //
  // `NaN` is the chosen sentinel for both modes:
  //   * Regression: no float is forbidden as a valid prediction, so `NaN`
  //     is the only value guaranteed not to collide with real data.
  //   * Classification: group ids are integers; `NaN` survives the
  //     `OutcomeVector` (float) representation cleanly and can't be
  //     mistaken for a valid 0-based group id. (Earlier this used `-1`,
  //     which worked but required a separate filter path on every caller;
  //     unifying on `NaN` lets callers use `std::isnan` regardless of mode.)
  OutcomeVector oob_predict(Forest const& forest, FeatureMatrix const& x) {
    struct Visitor : Model::Visitor {
      FeatureMatrix const& x;
      OutcomeVector result;

      explicit Visitor(FeatureMatrix const& x)
          : x(x) {}

      void visit(Forest const&) override {
        throw UserError("oob_predict: unknown forest mode (expected classification or regression).");
      }
      void visit(ClassificationForest const& forest) override {
        result = scatter(oob_predictions_with(forest, x, majority_vote), x.rows());
      }
      void visit(RegressionForest const& forest) override {
        result = scatter(oob_predictions_with(forest, x, mean), x.rows());
      }

      static OutcomeVector scatter(OOBPredictions const& oob, Eigen::Index n) {
        OutcomeVector out(n);
        out.fill(std::numeric_limits<Outcome>::quiet_NaN());
        for (int k = 0; k < static_cast<int>(oob.rows.size()); ++k) {
          out(oob.rows[static_cast<std::size_t>(k)]) = oob.predictions(k);
        }
        return out;
      }
    };

    Visitor visitor(x);
    forest.accept(visitor);
    return std::move(visitor.result);
  }

  ClassificationMetrics::Maybe
  oob_metrics(ClassificationForest const& forest, FeatureMatrix const& x, OutcomeVector const& y) {
    OOBPredictions const oob = oob_predictions_with(forest, x, majority_vote);
    if (oob.rows.empty()) {
      return std::nullopt;
    }

    GroupIdVector const y_oob = y(oob.rows, Eigen::all).cast<GroupId>().eval();
    return ClassificationMetrics(oob.predictions.cast<GroupId>(), y_oob);
  }

  RegressionMetrics::Maybe oob_metrics(RegressionForest const& forest, FeatureMatrix const& x, OutcomeVector const& y) {
    user_error(
        y.size() == x.rows(),
        "oob_metrics: response length (" + std::to_string(y.size()) +
            ") does not match the number of observations in x (" + std::to_string(x.rows()) + ")."
    );

    OOBPredictions const oob = oob_predictions_with(forest, x, mean);
    if (oob.rows.empty()) {
      return std::nullopt;
    }

    OutcomeVector const y_oob = y(oob.rows, Eigen::all).eval();
    return RegressionMetrics(oob.predictions, y_oob);
  }

  std::optional<double> oob_error(Forest const& forest, FeatureMatrix const& x, OutcomeVector const& y) {
    struct Visitor : Model::Visitor {
      FeatureMatrix const& x;
      OutcomeVector const& y;
      std::optional<double> result;

      Visitor(FeatureMatrix const& x, OutcomeVector const& y)
          : x(x)
          , y(y) {}

      void visit(Forest const&) override {
        throw UserError("oob_error: unknown forest mode (expected classification or regression).");
      }
      void visit(ClassificationForest const& forest) override {
        if (auto m = oob_metrics(forest, x, y)) {
          result = m->error_rate();
        }
      }
      void visit(RegressionForest const& forest) override {
        if (auto m = oob_metrics(forest, x, y)) {
          result = m->mse;
        }
      }
    };

    Visitor visitor(x, y);
    forest.accept(visitor);
    return visitor.result;
  }

  std::optional<double> oob_error(Forest const& forest, FeatureMatrix const& x, GroupIdVector const& y) {
    return oob_error(forest, x, OutcomeVector(y.cast<Outcome>()));
  }

  double error(Model const& model, FeatureMatrix const& x, OutcomeVector const& y) {
    struct Visitor : Model::Visitor {
      FeatureMatrix const& x;
      OutcomeVector const& y;
      double result = 0.0;

      Visitor(FeatureMatrix const& x, OutcomeVector const& y)
          : x(x)
          , y(y) {}

      void visit(Tree const&) override {
        throw UserError("error: unknown model mode (expected classification or regression).");
      }
      void visit(Forest const&) override {
        throw UserError("error: unknown model mode (expected classification or regression).");
      }
      void visit(ClassificationTree const& m) override { result = stats::error_rate(m.predict(x), y); }
      void visit(ClassificationForest const& m) override { result = stats::error_rate(m.predict(x), y); }
      void visit(RegressionTree const& m) override { result = stats::mse(m.predict(x), y); }
      void visit(RegressionForest const& m) override { result = stats::mse(m.predict(x), y); }
    };

    Visitor visitor(x, y);
    model.accept(visitor);
    return visitor.result;
  }
}
