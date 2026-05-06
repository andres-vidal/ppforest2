#include "models/strategies/pp/PDA.hpp"

#include "models/strategies/NodeContext.hpp"
#include "utils/Invariant.hpp"
#include "utils/JsonReader.hpp"

#include <cmath>
#include <limits>
#include <Eigen/Dense>
#include <nlohmann/json.hpp>

using namespace ppforest2::types;
using namespace ppforest2::stats;

namespace ppforest2::pp {
  namespace {
    ProjectionPursuit::Result nan_result(FeatureMatrix const& x) {
      auto const projector   = FeatureVector::Constant(x.cols(), std::numeric_limits<Feature>::quiet_NaN());
      auto const index_value = std::numeric_limits<Feature>::quiet_NaN();
      return ProjectionPursuit::Result{projector, index_value};
    }

    ProjectionPursuit::Result optimize_pda(Feature lambda, FeatureMatrix const& x, GroupPartition const& y_part) {
      FeatureMatrix const B = y_part.bgss(x);
      FeatureMatrix const W = y_part.wgss(x);

      FeatureMatrix W_pda = (1 - lambda) * W;
      W_pda.diagonal()    = W.diagonal();

      FeatureMatrix const WpB = W_pda + B; // symmetric

      Eigen::GeneralizedSelfAdjointEigenSolver<FeatureMatrix> ges;
      ges.compute(B, WpB);

      if (ges.info() != Eigen::Success) {
        return nan_result(x);
      }

      // largest eigenvalue → best 1D projection
      FeatureVector const max_eigen_vec = ges.eigenvectors().col(ges.eigenvalues().size() - 1);
      Feature const max_eigen_val       = ges.eigenvalues().real().maxCoeff();

      // Solver may report Success but produce NaN eigenvectors when B is
      // positive-semidefinite (singular covariance from small bootstrap samples).
      if (max_eigen_vec.hasNaN() || std::isnan(max_eigen_val)) {
        return nan_result(x);
      }

      return ProjectionPursuit::Result{ppforest2::pp::normalize(max_eigen_vec), max_eigen_val};
    }
  }

  PDA::PDA(float lambda)
      : lambda(lambda) {}

  nlohmann::json PDA::to_json() const {
    return {{"name", "pda"}, {"lambda", lambda}};
  }

  void PDA::compute(NodeContext& ctx, stats::RNG& /*rng*/) const {
    invariant(ctx.var_selection.has_value(), "PDA requires var_selection on NodeContext");
    auto const& partition = ctx.active_partition();
    auto reduced_x        = ctx.x(Eigen::all, ctx.var_selection->selected_cols);
    auto result           = optimize_pda(lambda, reduced_x, partition);
    ctx.projector         = ctx.var_selection->expand(result.projector);
    ctx.pp_index_value    = result.index_value;
  }


  ProjectionPursuit::Ptr pda(float lambda) {
    return std::make_shared<PDA>(lambda);
  }

  ProjectionPursuit::Ptr PDA::from_json(nlohmann::json const& j) {
    JsonReader const r{j, "pda"};
    r.only_keys({"name", "lambda"});
    return pda(static_cast<float>(r.require_number("lambda", 0.0, 1.0)));
  }
}
