#include "ppforest2.hpp"

#include <RcppCommon.h>

using namespace ppforest2;
using namespace ppforest2::types;
using namespace ppforest2::stats;
using namespace ppforest2::serialization;
using namespace ppforest2::pp;

constexpr char const* CLASS_PPRF = "pprf";
constexpr char const* CLASS_PPTR = "pptr";

// Index conversion helpers: C++ uses 0-based indices, R uses 1-based.
inline Outcome to_r_index(Outcome i) {
  return i + 1;
}
inline Outcome to_cpp_index(Outcome i) {
  return i - 1;
}
inline void to_r_indices(OutcomeVector& v) {
  v.array() += 1;
}
inline void to_cpp_indices(OutcomeVector& v) {
  v.array() -= 1;
}

// R-side classification labels are 1-based factor codes; C++ uses 0-based
// GroupIds. For regression `y` is the continuous response and must be left
// alone. These helpers centralise the "shift if classification" decision
// at both ends of the Rcpp boundary so callers don't repeat the mode check.
//
//   * `to_cpp_indices_if_classification` decodes incoming `y` from R
//     (used on training, OOB-error, and VI entry points).
//   * `to_r_indices_if_classification` encodes outgoing predictions for
//     R (used on `predict_tree`, `predict_forest`, and `oob_predict`).
//     For regression the prediction vector is raw floats — sentinels
//     (NaN for OOB-predict) pass through unchanged.
inline void to_cpp_indices_if_classification(Model const& model, OutcomeVector& y) {
  if (is_classification(model)) {
    to_cpp_indices(y);
  }
}

inline void to_r_indices_if_classification(Model const& model, OutcomeVector& v) {
  if (is_classification(model)) {
    to_r_indices(v);
  }
}

namespace Rcpp {
  // Tree node / leaf / branch handling is mode-aware (see the
  // `wrap_with_mode` / `as_with_mode` helpers below): classification
  // leaf values and branch group ids get the +1 R-indexing shift,
  // regression leaf values (raw float predictions) pass through
  // unchanged. We expose only the model-level wraps publicly — the
  // generic single-arg `Rcpp::wrap(TreeLeaf const&)` etc. would have to
  // pick one mode's behaviour and be wrong for the other, so they're
  // intentionally not declared here.
  SEXP wrap(Tree const&);
  SEXP wrap(BaggedTree const&);
  SEXP wrap(Forest const&);
  SEXP wrap(Model::Ptr const&);

  SEXP wrap(TrainingSpec const&);
  SEXP wrap(VariableImportance const&);
  SEXP wrap(ppforest2::stats::ConfusionMatrix const&);
  SEXP wrap(ppforest2::stats::ClassificationMetrics const&);
  SEXP wrap(ppforest2::stats::RegressionMetrics const&);
  SEXP wrap(Export<Model::Ptr> const&);

  template<> Tree::Ptr as(SEXP);
  template<> std::unique_ptr<BaggedTree> as(SEXP);
  template<> Forest::Ptr as(SEXP);

  template<> Model::Ptr as(SEXP);
  template<> TrainingSpec::Ptr as(SEXP);
}


#include <Rcpp.h>

// R-side classification models carry `model$groups` as a character vector
// of label names (e.g. `c("setosa", "versicolor", "virginica")`); the
// canonical `ClassificationForest::groups` / `ClassificationTree::groups`
// in C++ is `std::set<GroupId>` — the 0-based ascending indices into that
// names vector. This helper performs the cross-boundary derivation.
//
// The C++ side never *writes* a "groups" field into a wrap; R fills the
// names in itself after training. We only need to read names back when
// converting an R model to C++.
inline std::set<types::GroupId> group_ids_from_names(Rcpp::CharacterVector const& group_names) {
  std::set<types::GroupId> ids;
  for (int i = 0; i < group_names.size(); ++i) {
    ids.insert(static_cast<types::GroupId>(i));
  }
  return ids;
}

namespace Rcpp {
  // Mode-aware tree-node wrapping. Leaf values and branch group ids are
  // 0-based group ids for classification (shift +1 here to match R's
  // 1-based factor indexing) and raw float predictions / artificial
  // 2-group ids for regression (passed through unchanged — regression
  // leaves are continuous responses, not indices).
  //
  // Each helper threads `mode` explicitly so the shift decision is
  // visible at the call site instead of buried in a generic wrap.
  SEXP wrap_node_with_mode(TreeNode const& node, types::Mode mode);

  inline SEXP wrap_leaf_with_mode(TreeLeaf const& node, types::Mode mode) {
    Outcome const value = is_classification(mode) ? to_r_index(node.value) : node.value;
    Rcpp::List result   = Rcpp::List::create(Rcpp::Named("value") = Rcpp::wrap(value));
    if (node.degenerate) {
      result["degenerate"] = true;
    }
    return result;
  }

  inline SEXP wrap_branch_with_mode(TreeBranch const& node, types::Mode mode) {
    Rcpp::IntegerVector groups(node.groups.begin(), node.groups.end());
    if (is_classification(mode)) {
      for (int k = 0; k < groups.size(); ++k) {
        groups[k] = to_r_index(groups[k]);
      }
    }
    Rcpp::List result = Rcpp::List::create(
        Rcpp::Named("projector")      = Rcpp::wrap(node.projector),
        Rcpp::Named("cutpoint")       = Rcpp::wrap(node.cutpoint),
        Rcpp::Named("pp_index_value") = Rcpp::wrap(node.pp_index_value),
        Rcpp::Named("groups")         = groups,
        Rcpp::Named("lower")          = wrap_node_with_mode(*node.lower, mode),
        Rcpp::Named("upper")          = wrap_node_with_mode(*node.upper, mode)
    );
    if (node.degenerate) {
      result["degenerate"] = true;
    }
    return result;
  }

  inline SEXP wrap_node_with_mode(TreeNode const& node, types::Mode mode) {
    struct ModeAwareWrapper : public TreeNode::Visitor {
      types::Mode mode;
      SEXP result;
      explicit ModeAwareWrapper(types::Mode m)
          : mode(m)
          , result(R_NilValue) {}
      void visit(TreeBranch const& b) override { result = wrap_branch_with_mode(b, mode); }
      void visit(TreeLeaf const& l) override { result = wrap_leaf_with_mode(l, mode); }
    };
    ModeAwareWrapper wrapper(mode);
    node.accept(wrapper);
    return wrapper.result;
  }

  // Build the full 3-level S3 class vector for an R-side ppforest2 model:
  //   c("{pptr,pprf}_<mode>", "{pptr,pprf}", "ppmodel")
  // Owned by the wrap layer because the C++ side is the authoritative source
  // for `mode` (it lives on the TrainingSpec). Centralising here removes
  // the same logic from `pptr()`, `pprf()`, and `load_json()` on the R side.
  inline Rcpp::CharacterVector make_model_class(char const* parent, types::Mode mode) {
    return Rcpp::CharacterVector::create(std::string(parent) + "_" + types::to_string(mode), parent, "ppmodel");
  }

  inline SEXP wrap(Tree const& tree) {
    Rcpp::List result = Rcpp::List::create(
        Rcpp::Named("training_spec") = Rcpp::wrap(*tree.training_spec),
        Rcpp::Named("root")          = wrap_node_with_mode(*tree.root, tree.training_spec->mode),
        Rcpp::Named("degenerate")    = tree.degenerate
    );
    result.attr("class") = make_model_class(CLASS_PPTR, tree.training_spec->mode);
    return result;
  }

  inline SEXP wrap(BaggedTree const& tree) {
    Rcpp::List result = Rcpp::List::create(
        Rcpp::Named("training_spec")  = Rcpp::wrap(*tree.model->training_spec),
        Rcpp::Named("root")           = wrap_node_with_mode(*tree.model->root, tree.model->training_spec->mode),
        Rcpp::Named("sample_indices") = Rcpp::wrap(tree.sample_indices)
    );
    result.attr("class") = make_model_class(CLASS_PPTR, tree.model->training_spec->mode);
    return result;
  }

  inline SEXP wrap(Forest const& forest) {
    Rcpp::List trees(forest.trees.size());

    for (size_t i = 0; i < forest.trees.size(); i++) {
      trees[i] = wrap(*forest.trees[i]);
    }

    Rcpp::List result = Rcpp::List::create(
        Rcpp::Named("training_spec") = Rcpp::wrap(*forest.training_spec),
        Rcpp::Named("trees")         = trees,
        Rcpp::Named("degenerate")    = forest.degenerate
    );
    result.attr("class") = make_model_class(CLASS_PPRF, forest.training_spec->mode);
    return result;
  }

  inline SEXP wrap(Model::Ptr const& model) {
    struct WrapVisitor : public Model::Visitor {
      SEXP result;
      void visit(Tree const& tree) override { result = Rcpp::wrap(tree); }
      void visit(Forest const& forest) override { result = Rcpp::wrap(forest); }
    };

    WrapVisitor visitor;
    model->accept(visitor);
    return visitor.result;
  }

  inline SEXP wrap(ppforest2::stats::ConfusionMatrix const& cm) {
    int const n = static_cast<int>(cm.values.rows());
    Rcpp::IntegerMatrix matrix(n, n);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        matrix(i, j) = cm.values(i, j);
      }
    }

    Rcpp::IntegerVector labels(cm.label_index.size());
    int idx = 0;
    for (auto const& [label, _] : cm.label_index) {
      labels[idx++] = label;
    }

    auto const ge = cm.group_errors();
    Rcpp::NumericVector group_errors(ge.size());
    for (Eigen::Index i = 0; i < ge.size(); ++i) {
      group_errors[i] = ge(i);
    }

    return Rcpp::List::create(
        Rcpp::Named("matrix") = matrix, Rcpp::Named("labels") = labels, Rcpp::Named("group_errors") = group_errors
    );
  }

  inline SEXP wrap(ppforest2::stats::ClassificationMetrics const& cm) {
    return Rcpp::List::create(
        Rcpp::Named("confusion_matrix") = Rcpp::wrap(cm.confusion_matrix), Rcpp::Named("error_rate") = cm.error_rate()
    );
  }

  inline SEXP wrap(ppforest2::stats::RegressionMetrics const& rm) {
    return Rcpp::List::create(
        Rcpp::Named("mse") = rm.mse, Rcpp::Named("mae") = rm.mae, Rcpp::Named("r_squared") = rm.r_squared
    );
  }

  inline SEXP wrap(Export<Model::Ptr> const& e) {
    Rcpp::List result = Rcpp::wrap(e.model);

    // `e.groups` is empty for regression (no group concept) — R sees an
    // empty character vector either way.
    result["groups"] = Rcpp::wrap(e.groups);
    if (e.model->training_spec)
      result["seed"] = e.model->training_spec->seed;
    if (e.variable_importance)
      result["vi"] = Rcpp::wrap(*e.variable_importance);

    // Mirror the C++ Export shape: full ClassificationMetrics or
    // RegressionMetrics block per mode (not a flattened scalar). R-side
    // accessors like `oob_error()` derive the scalar from this block,
    // so the JSON payload that travelled through C++ → R is preserved.
    auto wrap_metrics = [](serialization::Metrics const& m) -> SEXP {
      return std::visit([](auto const& alt) -> SEXP { return Rcpp::wrap(alt); }, m);
    };
    if (e.training_metrics)
      result["training_metrics"] = wrap_metrics(*e.training_metrics);
    if (e.oob_metrics)
      result["oob_metrics"] = wrap_metrics(*e.oob_metrics);

    // Re-apply the full 3-level S3 class. The inner `wrap(Model::Ptr)`
    // already set it via `make_model_class`, but the subsequent
    // `result["..."] = ...` field assignments in this function strip the
    // class attribute as a side-effect of how `Rcpp::List::operator[]`
    // rebuilds the underlying named-list. Recomputing via the same
    // helper keeps the wrap layer the single source of truth on class.
    struct ModeVisitor : public Model::Visitor {
      types::Mode mode;
      char const* parent;
      void visit(Tree const& t) override {
        mode   = t.training_spec->mode;
        parent = CLASS_PPTR;
      }
      void visit(Forest const& f) override {
        mode   = f.training_spec->mode;
        parent = CLASS_PPRF;
      }
    };
    ModeVisitor mv;
    e.model->accept(mv);
    result.attr("class") = make_model_class(mv.parent, mv.mode);

    return result;
  }

  /**
   * Convert a flat JSON object to an Rcpp named list.
   *
   * Supports string, number (int/float), and boolean values.
   * Used to generically wrap strategy JSON so that adding a new
   * strategy only requires implementing to_json() — no changes here.
   */
  inline Rcpp::List json_to_list(nlohmann::json const& j) {
    Rcpp::List result;

    for (auto& [key, val] : j.items()) {
      if (val.is_string()) {
        result[key] = val.get<std::string>();
      } else if (val.is_number_integer()) {
        result[key] = val.get<int>();
      } else if (val.is_number_float()) {
        result[key] = val.get<float>();
      } else if (val.is_boolean()) {
        result[key] = val.get<bool>();
      } else if (val.is_object()) {
        result[key] = json_to_list(val);
      } else if (val.is_array()) {
        // Array of (presumably) objects — used by composite strategies.
        Rcpp::List arr(val.size());
        for (std::size_t k = 0; k < val.size(); ++k) {
          arr[k] = val[k].is_object() ? json_to_list(val[k]) : Rcpp::List();
        }
        result[key] = arr;
      }
    }
    return result;
  }

  /**
   * Convert an Rcpp named list to a JSON object.
   *
   * Inverse of json_to_list(). Supports string, integer, real, boolean,
   * nested lists (objects), and unnamed lists (arrays of objects) —
   * the latter is needed for composite strategies like stop_any whose
   * `rules` field is a list of stop rules.
   */
  inline nlohmann::json list_to_json(Rcpp::List const& list) {
    nlohmann::json j;
    Rcpp::CharacterVector names = list.names();

    for (int i = 0; i < list.size(); i++) {
      std::string key = Rcpp::as<std::string>(names[i]);

      // display_name is a presentation-only field, not part of serialization.
      if (key == "display_name")
        continue;

      switch (TYPEOF(list[i])) {
        case STRSXP: j[key] = Rcpp::as<std::string>(list[i]); break;
        case INTSXP: j[key] = Rcpp::as<int>(list[i]); break;
        case REALSXP: j[key] = Rcpp::as<float>(list[i]); break;
        case LGLSXP: j[key] = Rcpp::as<bool>(list[i]); break;
        case VECSXP: {
          // Nested list. If it has names, treat as object; otherwise as array.
          Rcpp::List sub(list[i]);
          Rcpp::RObject sub_names_obj = sub.names();

          if (sub_names_obj.isNULL()) {
            // Unnamed list → JSON array. Each element is recursively a list.
            nlohmann::json arr = nlohmann::json::array();
            for (int k = 0; k < sub.size(); ++k) {
              if (TYPEOF(sub[k]) == VECSXP) {
                arr.push_back(list_to_json(Rcpp::List(sub[k])));
              }
            }
            j[key] = arr;
          } else {
            j[key] = list_to_json(sub);
          }
          break;
        }
        default: break;
      }
    }
    return j;
  }

  /**
   * Convert a TrainingSpec to an R list.
   *
   * C++ TrainingSpec → JSON → R list: uses the strategy registry so new
   * strategies work here automatically without changes to this file.
   */
  /** Convert a strategy shared_ptr to an R list (JSON + display_name). */
  template<typename T, std::enable_if_t<std::is_base_of_v<Strategy<T>, T>, int> = 0>
  inline SEXP wrap(std::shared_ptr<T> const& strategy) {
    auto j            = strategy->to_json();
    j["display_name"] = strategy->display_name();
    return json_to_list(j);
  }

  inline SEXP wrap(TrainingSpec const& spec) {
    std::string const mode_str = is_regression(spec) ? "regression" : "classification";

    return Rcpp::List::create(
        Rcpp::Named("pp")          = wrap(spec.pp),
        Rcpp::Named("vars")        = wrap(spec.vars),
        Rcpp::Named("cutpoint")    = wrap(spec.cutpoint),
        Rcpp::Named("stop")        = wrap(spec.stop),
        Rcpp::Named("binarize")    = wrap(spec.binarization),
        Rcpp::Named("grouping")    = wrap(spec.grouping),
        Rcpp::Named("leaf")        = wrap(spec.leaf),
        Rcpp::Named("mode")        = mode_str,
        Rcpp::Named("size")        = spec.size,
        Rcpp::Named("seed")        = spec.seed,
        Rcpp::Named("threads")     = spec.threads,
        Rcpp::Named("max_retries") = spec.max_retries
    );
  }

  inline SEXP wrap(VariableImportance const& vi) {
    Rcpp::List list;
    if (vi.scale.size() > 0)
      list["scale"] = Rcpp::wrap(vi.scale);
    if (vi.projections.size() > 0)
      list["projections"] = Rcpp::wrap(vi.projections);
    if (vi.weighted_projections.size() > 0)
      list["weighted"] = Rcpp::wrap(vi.weighted_projections);
    if (vi.permuted.size() > 0)
      list["permuted"] = Rcpp::wrap(vi.permuted);
    return list;
  }

  // Mode-aware tree-node reconstruction. Inverse of `wrap_node_with_mode`:
  // classification leaf values / branch group ids are decoded from
  // R's 1-based factor indices back to 0-based group ids; regression
  // leaf values pass through unchanged.
  inline std::unique_ptr<TreeNode> as_node_with_mode(SEXP x, types::Mode mode) {
    Rcpp::List rnode(x);

    if (rnode.containsElementNamed("value")) {
      Feature const raw   = Rcpp::as<Feature>(rnode["value"]);
      Feature const value = is_classification(mode) ? to_cpp_index(raw) : raw;
      auto leaf           = std::make_unique<TreeLeaf>(value);
      leaf->degenerate    = rnode.containsElementNamed("degenerate") && Rcpp::as<bool>(rnode["degenerate"]);
      return leaf;
    }

    auto lower = as_node_with_mode(rnode["lower"], mode);
    auto upper = as_node_with_mode(rnode["upper"], mode);

    std::set<GroupId> groups;
    if (rnode.containsElementNamed("groups")) {
      Rcpp::IntegerVector rgroups(rnode["groups"]);
      for (auto g : rgroups) {
        groups.insert(is_classification(mode) ? to_cpp_index(g) : g);
      }
    }

    Feature pp_index_value = 0;
    if (rnode.containsElementNamed("pp_index_value")) {
      pp_index_value = Rcpp::as<Feature>(rnode["pp_index_value"]);
    }

    return TreeBranch::make(
        Rcpp::as<Projector>(rnode["projector"]),
        Rcpp::as<Feature>(rnode["cutpoint"]),
        std::move(lower),
        std::move(upper),
        std::move(groups),
        pp_index_value
    );
  }

  // Construct the concrete Tree subclass based on the spec's mode.
  template<> inline Tree::Ptr as(SEXP x) {
    Rcpp::List rtree(x);

    auto spec = as<TrainingSpec::Ptr>(rtree["training_spec"]);
    auto root = as_node_with_mode(rtree["root"], spec->mode);

    if (is_regression(spec)) {
      return std::make_unique<RegressionTree>(std::move(root), spec);
    }

    // Classification trees in R carry their canonical label set under
    // `tree$groups` (a character vector of names, populated by the R
    // training/loading code — see `pprf.R` and `json.R`). Derive 0-based
    // ids from its length. Required — `ClassificationTree`'s constructor
    // asserts non-empty.
    Rcpp::CharacterVector group_names = rtree["groups"];
    auto groups                       = group_ids_from_names(group_names);
    return std::make_unique<ClassificationTree>(std::move(root), spec, std::move(groups));
  }

  template<> inline std::unique_ptr<BaggedTree> as(SEXP x) {
    Rcpp::List rtree(x);

    auto inner = as<Tree::Ptr>(x); // uses the mode in training_spec
    return BaggedTree::make(std::move(inner), as<std::vector<int>>(rtree["sample_indices"]));
  }

  template<> inline Forest::Ptr as(SEXP x) {
    Rcpp::List rforest(x);
    Rcpp::List rtrees(rforest["trees"]);
    Rcpp::List rtraining_spec(rforest["training_spec"]);

    auto spec = as<TrainingSpec::Ptr>(rtraining_spec);

    // Mirrors `meta.groups` in the JSON path: classification forests carry
    // the canonical training label set under `forest$groups` (a character
    // vector of names, populated R-side after training). Derive 0-based
    // ids from its length. Required for `predict(*, Proportions)` to
    // produce the correct column layout, and asserted non-empty by the
    // `ClassificationForest` constructor.
    Forest::Ptr forest;
    if (is_regression(spec)) {
      forest = std::make_unique<RegressionForest>(spec);
    } else {
      Rcpp::CharacterVector group_names = rforest["groups"];
      auto groups                       = group_ids_from_names(group_names);
      forest                            = std::make_unique<ClassificationForest>(spec, std::move(groups));
    }

    for (size_t i = 0; i < rtrees.size(); ++i) {
      auto bt = as<std::unique_ptr<BaggedTree>>(rtrees[i]);
      forest->add_tree(std::move(bt));
    }

    return forest;
  }

  template<> inline Model::Ptr as(SEXP x) {
    if (Rcpp::RObject(x).inherits(CLASS_PPRF)) {
      return std::shared_ptr<Forest>(as<Forest::Ptr>(x).release());
    } else {
      return std::shared_ptr<Tree>(as<Tree::Ptr>(x).release());
    }
  }

  /**
   * Convert an R list to a TrainingSpec.
   *
   * R list → JSON → C++ strategy. Strategy entries are optional: when
   * an entry is missing or `NULL`, the Builder fills in the mode-aware
   * default via `apply_defaults()` at `make()` time. R-side helpers
   * (`resolve_strategies` in `util.R`) therefore only need to populate
   * the strategies the user explicitly supplied.
   */
  template<> inline TrainingSpec::Ptr as(SEXP x) {
    Rcpp::List r_training_spec(x);

    types::Mode mode = types::Mode::Classification;
    if (r_training_spec.containsElementNamed("mode")) {
      std::string mode_str = Rcpp::as<std::string>(r_training_spec["mode"]);
      if (mode_str == "regression") {
        mode = types::Mode::Regression;
      }
    }

    auto builder = TrainingSpec::builder(mode);
    builder.size(Rcpp::as<int>(r_training_spec["size"]))
        .seed(Rcpp::as<int>(r_training_spec["seed"]))
        .threads(Rcpp::as<int>(r_training_spec["threads"]))
        .max_retries(Rcpp::as<int>(r_training_spec["max_retries"]));

    // Each strategy is optional. Skip slots the caller didn't populate
    // (missing key or explicit `NULL` on the R side) — `Builder::build()`
    // calls `apply_defaults()` to fill them in with mode-aware values.
    auto has_strategy = [&](char const* key) {
      return r_training_spec.containsElementNamed(key) && !Rf_isNull(r_training_spec[key]);
    };

    if (has_strategy("pp")) {
      builder.pp(pp::ProjectionPursuit::from_json(list_to_json(Rcpp::List(r_training_spec["pp"]))));
    }
    if (has_strategy("vars")) {
      builder.vars(vars::VariableSelection::from_json(list_to_json(Rcpp::List(r_training_spec["vars"]))));
    }
    if (has_strategy("cutpoint")) {
      builder.cutpoint(cutpoint::Cutpoint::from_json(list_to_json(Rcpp::List(r_training_spec["cutpoint"]))));
    }
    if (has_strategy("stop")) {
      builder.stop(stop::StopRule::from_json(list_to_json(Rcpp::List(r_training_spec["stop"]))));
    }
    if (has_strategy("binarize")) {
      builder.binarization(binarize::Binarization::from_json(list_to_json(Rcpp::List(r_training_spec["binarize"]))));
    }
    if (has_strategy("grouping")) {
      builder.grouping(grouping::Grouping::from_json(list_to_json(Rcpp::List(r_training_spec["grouping"]))));
    }
    if (has_strategy("leaf")) {
      builder.leaf(leaf::LeafStrategy::from_json(list_to_json(Rcpp::List(r_training_spec["leaf"]))));
    }

    return builder.make();
  }
}
