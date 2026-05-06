#include "../inst/include/ppforest2.h"
#include <RcppEigen.h>
#include "serialization/Json.hpp"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <fstream>

// [[Rcpp::depends(RcppEigen)]]

using namespace Rcpp;
using namespace RcppEigen;
using namespace ppforest2;
using namespace ppforest2::types;
using namespace ppforest2::stats;
using namespace ppforest2::pp;
using namespace ppforest2::viz;
using namespace ppforest2::serialization;

// [[Rcpp::export]]
bool ppforest2_has_openmp() {
#ifdef _OPENMP
  return true;
#else
  return false;
#endif
}

// Mode-aware Rcpp wrappers. Each function takes a TrainingSpec or a
// trained Model and uses its `mode` field to branch — callers on the R
// side never have to know which mode they're in. The only mode-specific
// concerns at this boundary are:
//   * classification needs `to_cpp_indices(y)` on entry (R is 1-based,
//     C++ is 0-based) and `to_r_indices(result)` on outgoing predictions;
//     regression skips both since `y` is a continuous response;
//   * classification expects `y` to be group-contiguous before training
//     (ClassificationTree's invariant); regression expects `y` sorted
//     ascending (ByCutpoint's `compute_init` contract).

// [[Rcpp::export]]
Model::Ptr ppforest2_train(TrainingSpec::Ptr spec, FeatureMatrix x, OutcomeVector y) {
  if (is_classification(*spec)) {
    to_cpp_indices(y);

    // y carries integer class labels as float; sort cast-to-int and then
    // re-cast so `x` and `y` stay in lockstep through the sort.
    GroupIdVector y_int = y.cast<GroupId>();
    if (!GroupPartition::is_contiguous(y_int)) {
      sort(x, y_int);
    }

    OutcomeVector y_out = y_int.cast<Outcome>();
    return Model::train(*spec, x, y_out);
  }

  // Regression: ByCutpoint's median split needs y-sorted rows. The R-side
  // `resolve_model_data` already sorts, but mirror the classification
  // defensive sort so any caller that forgets is still correct.
  if (!std::is_sorted(y.data(), y.data() + y.size())) {
    sort(x, y);
  }

  return Model::train(*spec, x, y);
}

// [[Rcpp::export]]
OutcomeVector ppforest2_predict_tree(Tree::Ptr const& tree, FeatureMatrix const& data) {
  OutcomeVector result = tree->predict(data);
  to_r_indices_if_classification(*tree, result);
  return result;
}

// [[Rcpp::export]]
OutcomeVector ppforest2_predict_forest(Forest::Ptr const& forest, FeatureMatrix const& data) {
  OutcomeVector result = forest->predict(data);
  to_r_indices_if_classification(*forest, result);
  return result;
}

// [[Rcpp::export]]
FeatureMatrix ppforest2_predict_tree_prob(Tree::Ptr const& tree, FeatureMatrix const& data) {
  return predict_proportions(*tree, data);
}

// [[Rcpp::export]]
FeatureMatrix ppforest2_predict_forest_prob(Forest::Ptr const& forest, FeatureMatrix const& data) {
  return predict_proportions(*forest, data);
}

// [[Rcpp::export]]
FeatureVector ppforest2_vi_projections_tree(Tree::Ptr const& tree, int n_vars, FeatureVector const& scale) {
  return vi_projections(tree, n_vars, &scale);
}

// [[Rcpp::export]]
FeatureVector ppforest2_vi_projections_forest(Forest::Ptr const& forest, int n_vars, FeatureVector const& scale) {
  return vi_projections(forest, n_vars, &scale);
}

// [[Rcpp::export]]
FeatureVector ppforest2_vi_weighted_forest(
    Forest::Ptr const& forest, FeatureMatrix const& x, OutcomeVector y, FeatureVector const& scale
) {
  to_cpp_indices_if_classification(*forest, y);
  return vi_weighted_projections(forest, x, y, &scale);
}

// [[Rcpp::export]]
FeatureVector
ppforest2_vi_permuted_forest(Forest::Ptr const& forest, FeatureMatrix const& x, OutcomeVector y, int seed) {
  to_cpp_indices_if_classification(*forest, y);
  return vi_permuted(forest, x, y, seed);
}

namespace {
  // Translate std::optional<double> into an R-visible scalar: a length-1
  // NumericVector carrying the value, or `NA_real_` when no OOB data was
  // available. This is the one well-defined way R callers can see
  // "missing" — a plain `double` return value can't express NA.
  Rcpp::NumericVector to_r_scalar(std::optional<double> const& x) {
    if (x) {
      return Rcpp::NumericVector::create(*x);
    } else {
      return Rcpp::NumericVector::create(NA_REAL);
    }
  }
}

// [[Rcpp::export]]
Rcpp::NumericVector ppforest2_oob_error(Forest::Ptr const& forest, FeatureMatrix const& x, OutcomeVector y) {
  to_cpp_indices_if_classification(*forest, y);
  return to_r_scalar(oob_error(forest, x, y));
}

// [[Rcpp::export]]
OutcomeVector ppforest2_oob_predict(Forest::Ptr const& forest, FeatureMatrix const& x) {
  // "No OOB tree" sentinel is `NaN` in both modes (see the comment on
  // `oob_predict` in `Evaluation.cpp`). For classification the +1 shift
  // applied below converts 0-based C++ group ids to 1-based R factor
  // indices; `NaN + 1` is still `NaN`, so the sentinel survives the
  // shift unchanged. The R-side accessors remap `NaN → NA_real_`
  // (regression) or `NaN → NA_real_` then to factor `NA` (classification).
  OutcomeVector result = oob_predict(forest, x);
  to_r_indices_if_classification(*forest, result);
  return result;
}

// [[Rcpp::export]]
Rcpp::List ppforest2_tree_node_data(Tree::Ptr const& tree, FeatureMatrix const& x, OutcomeVector y) {
  // Mode-aware: classification y is 1-based factor codes (decode to
  // 0-based GroupIds), regression y is a continuous response (leave
  // alone). Same gate for the +1 R-index shift on leaf values and
  // branch group ids below.
  bool const cls = is_classification(*tree);
  if (cls) {
    to_cpp_indices(y);
  }
  NodeDataVisitor visitor(x, y);
  tree->root->accept(visitor);

  Rcpp::List result(visitor.nodes.size());

  for (std::size_t i = 0; i < visitor.nodes.size(); ++i) {
    auto const& nd = visitor.nodes[i];

    Rcpp::IntegerVector groups_r(nd.groups.begin(), nd.groups.end());
    if (cls) {
      for (int k = 0; k < groups_r.size(); ++k) {
        groups_r[k] = to_r_index(groups_r[k]);
      }
    }

    if (nd.is_leaf) {
      // Classification: leaf value is a 0-based group id → shift to
      // 1-based factor code for R. Regression: leaf value is a raw
      // mean response → leave alone.
      Outcome const leaf_value = cls ? to_r_index(nd.value) : nd.value;
      result[i]                = Rcpp::List::create(
          Rcpp::Named("is_leaf") = true,
          Rcpp::Named("depth")   = nd.depth,
          Rcpp::Named("value")   = leaf_value,
          Rcpp::Named("groups")  = groups_r
      );
    } else {
      result[i] = Rcpp::List::create(
          Rcpp::Named("is_leaf")   = false,
          Rcpp::Named("depth")     = nd.depth,
          Rcpp::Named("projector") = Rcpp::wrap(nd.projector),
          Rcpp::Named("cutpoint")  = nd.cutpoint,
          Rcpp::Named("projected") = Rcpp::NumericVector(nd.projected_values.begin(), nd.projected_values.end()),
          Rcpp::Named("groups")    = groups_r
      );
    }
  }

  return result;
}

namespace {
  std::vector<std::pair<int, types::Feature>>
  build_fixed_vars(Rcpp::IntegerVector const& var_indices, Rcpp::NumericVector const& fixed_values) {
    std::vector<std::pair<int, types::Feature>> fixed_vars;

    if (fixed_values.size() > 0) {
      int p = static_cast<int>(var_indices.size()) + static_cast<int>(fixed_values.size());
      std::set<int> used(var_indices.begin(), var_indices.end());
      int fv_idx = 0;

      for (int k = 0; k < p; ++k) {
        if (used.find(k) == used.end()) {
          fixed_vars.push_back({k, static_cast<types::Feature>(fixed_values[fv_idx++])});
        }
      }
    }

    return fixed_vars;
  }
}

// [[Rcpp::export]]
Rcpp::DataFrame ppforest2_boundary_segments(
    Tree::Ptr const& tree,
    Rcpp::IntegerVector var_indices,
    Rcpp::NumericVector fixed_values,
    double x_min,
    double x_max,
    double y_min,
    double y_max
) {
  auto fixed_vars = build_fixed_vars(var_indices, fixed_values);

  BoundaryVisitor visitor(
      var_indices[0],
      var_indices[1],
      fixed_vars,
      static_cast<types::Feature>(x_min),
      static_cast<types::Feature>(x_max),
      static_cast<types::Feature>(y_min),
      static_cast<types::Feature>(y_max)
  );

  tree->root->accept(visitor);

  int n = static_cast<int>(visitor.segments.size());
  Rcpp::NumericVector xs(n), ys(n), xe(n), ye(n);
  Rcpp::IntegerVector depths(n);
  int idx = 0;

  for (auto const& seg : visitor.segments) {
    xs[idx]     = seg.x_start;
    ys[idx]     = seg.y_start;
    xe[idx]     = seg.x_end;
    ye[idx]     = seg.y_end;
    depths[idx] = seg.depth;
    idx++;
  }

  return Rcpp::DataFrame::create(
      Rcpp::Named("x_start") = xs,
      Rcpp::Named("y_start") = ys,
      Rcpp::Named("x_end")   = xe,
      Rcpp::Named("y_end")   = ye,
      Rcpp::Named("depth")   = depths
  );
}

// [[Rcpp::export]]
Rcpp::List ppforest2_decision_regions(
    Tree::Ptr const& tree,
    Rcpp::IntegerVector var_indices,
    Rcpp::NumericVector fixed_values,
    double x_min,
    double x_max,
    double y_min,
    double y_max
) {
  auto fixed_vars = build_fixed_vars(var_indices, fixed_values);

  RegionVisitor visitor(
      var_indices[0],
      var_indices[1],
      fixed_vars,
      static_cast<types::Feature>(x_min),
      static_cast<types::Feature>(x_max),
      static_cast<types::Feature>(y_min),
      static_cast<types::Feature>(y_max)
  );

  tree->root->accept(visitor);

  Rcpp::List result(visitor.regions.size());

  for (std::size_t i = 0; i < visitor.regions.size(); ++i) {
    auto const& region = visitor.regions[i];

    Rcpp::NumericVector rx(region.vertices.size());
    Rcpp::NumericVector ry(region.vertices.size());

    for (std::size_t j = 0; j < region.vertices.size(); ++j) {
      rx[j] = region.vertices[j].first;
      ry[j] = region.vertices[j].second;
    }

    result[i] = Rcpp::List::create(
        Rcpp::Named("x") = rx, Rcpp::Named("y") = ry, Rcpp::Named("group") = to_r_index(region.predicted_group)
    );
  }

  return result;
}

// [[Rcpp::export]]
Rcpp::List ppforest2_tree_layout(Tree::Ptr const& tree) {
  LayoutParams params;
  TreeLayout layout = compute_tree_layout(*tree->root, params);

  // Build node data frame
  int n_nodes = static_cast<int>(layout.nodes.size());
  Rcpp::NumericVector nx(n_nodes), ny(n_nodes);
  Rcpp::LogicalVector n_leaf(n_nodes);
  Rcpp::IntegerVector n_idx(n_nodes);
  int ni = 0;

  for (auto const& node : layout.nodes) {
    nx[ni]     = node.x;
    ny[ni]     = node.y;
    n_leaf[ni] = node.is_leaf;
    n_idx[ni]  = node.node_idx;
    ni++;
  }

  Rcpp::DataFrame node_df = Rcpp::DataFrame::create(
      Rcpp::Named("x") = nx, Rcpp::Named("y") = ny, Rcpp::Named("is_leaf") = n_leaf, Rcpp::Named("node_idx") = n_idx
  );

  // Build edge data frame
  int n_edges = static_cast<int>(layout.edges.size());
  Rcpp::NumericVector efx(n_edges), efy(n_edges), etx(n_edges), ety(n_edges);
  Rcpp::CharacterVector elabel(n_edges);
  int ei = 0;

  for (auto const& edge : layout.edges) {
    efx[ei]    = edge.from_x;
    efy[ei]    = edge.from_y;
    etx[ei]    = edge.to_x;
    ety[ei]    = edge.to_y;
    elabel[ei] = edge.label;
    ei++;
  }

  Rcpp::DataFrame edge_df = Rcpp::DataFrame::create(
      Rcpp::Named("from_x")     = efx,
      Rcpp::Named("from_y")     = efy,
      Rcpp::Named("to_x")       = etx,
      Rcpp::Named("to_y")       = ety,
      Rcpp::Named("edge_label") = elabel
  );

  return Rcpp::List::create(Rcpp::Named("nodes") = node_df, Rcpp::Named("edges") = edge_df);
}

namespace {
  using json = nlohmann::json;
}

// [[Rcpp::export]]
std::string ppforest2_save_model_json(
    Model::Ptr model,
    types::Names groups,
    bool include_metrics,
    FeatureMatrix const& x,
    OutcomeVector y,
    types::Names feature_names
) {
  if (include_metrics) {
    to_cpp_indices_if_classification(*model, y);
  }

  Export<Model::Ptr> model_export{
      std::move(model),
      std::move(groups), // empty for regression, non-empty for classification
      nullptr,
      static_cast<int>(x.rows()),
      static_cast<int>(x.cols()),
      std::move(feature_names),
  };

  if (include_metrics) {
    model_export.compute_metrics(x, y);
  }

  return model_export.to_json().dump(2);
}

// [[Rcpp::export]]
ppforest2::serialization::Export<Model::Ptr> ppforest2_load_model_json(std::string const& path) {
  std::ifstream in(path);

  if (!in.is_open()) {
    Rcpp::stop("Could not open file: " + path);
  }

  auto j = json::parse(in);
  return j.get<Export<Model::Ptr>>();
}
