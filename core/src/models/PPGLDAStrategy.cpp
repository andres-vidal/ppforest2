#include "models/PPGLDAStrategy.hpp"

#include "utils/Math.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
#include <Eigen/Dense>

using namespace ppforest2::types;
using namespace ppforest2::stats;


namespace ppforest2::pp {
  PPGLDAStrategy::PPGLDAStrategy(float lambda) :
    lambda(lambda) {
  }

  PPStrategy::Ptr PPGLDAStrategy::clone() const {
    return std::make_unique<PPGLDAStrategy>(*this);
  }

  Feature PPGLDAStrategy::index(
    FeatureMatrix const&  x,
    GroupPartition const& group_spec,
    Projector const&      projector) const {
    FeatureMatrix A = projector;

    FeatureMatrix W      = group_spec.wgss(x);
    FeatureMatrix W_diag = W.diagonal().asDiagonal();
    FeatureMatrix W_pda  = W_diag + (1 - lambda) * (W - W_diag);
    FeatureMatrix B      = group_spec.bgss(x);
    FeatureMatrix WpB    = W_pda + B;

    Feature denominator = (A.transpose() * WpB * A).determinant();

    if (std::fabs(denominator) < 1e-15) {
      return Feature(0);
    }

    Feature numerator = (A.transpose() * W_pda * A).determinant();

    return Feature(1) - numerator / denominator;
  }

  PPResult PPGLDAStrategy::optimize(
    const FeatureMatrix&  x,
    const GroupPartition& group_spec) const {
    const FeatureMatrix B = group_spec.bgss(x);
    const FeatureMatrix W = group_spec.wgss(x);

    FeatureMatrix W_pda = (Feature(1) - Feature(lambda)) * W;
    W_pda.diagonal() = W.diagonal();

    const FeatureMatrix WpB = W_pda + B; // symmetric

    Eigen::GeneralizedSelfAdjointEigenSolver<FeatureMatrix> ges;
    ges.compute(B, WpB);

    // largest eigenvalue => best 1D projection
    FeatureVector max_eigen_vec = ges.eigenvectors().col(ges.eigenvalues().size() - 1);
    Feature max_eigen_val       = ges.eigenvalues().real().maxCoeff();
    return PPResult{ pp::normalize(max_eigen_vec), max_eigen_val };
  }

  PPStrategy::Ptr PPGLDAStrategy::make(float lambda) {
    return std::make_unique<PPGLDAStrategy>(lambda);
  }

  PPStrategy::Ptr glda(float lambda) {
    return PPGLDAStrategy::make(lambda);
  }
}
