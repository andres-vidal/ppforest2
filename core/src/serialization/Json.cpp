#include "serialization/Json.hpp"
#include "models/ClassificationForest.hpp"
#include "models/ClassificationTree.hpp"
#include "models/Projector.hpp"

#include "models/Evaluation.hpp"
#include "models/RegressionForest.hpp"
#include "models/RegressionTree.hpp"
#include "models/TrainingSpec.hpp"
#include "serialization/ExportValidation.hpp"
#include "serialization/JsonOptional.hpp"
#include "stats/ConfusionMatrix.hpp"
#include "stats/Metrics.hpp"
#include "utils/Invariant.hpp"

#include <Eigen/Dense>
#include <algorithm>

using namespace ppforest2::types;
using namespace ppforest2::stats;


namespace ppforest2::serialization {

  namespace {
    types::GroupId to_group_id(types::Outcome const& id) {
      return static_cast<types::GroupId>(id);
    }

    types::Outcome to_outcome(types::GroupId id) {
      return static_cast<types::Outcome>(id);
    }

    std::string to_label(types::GroupId id, Names const& names) {
      return names[static_cast<std::size_t>(id)];
    }

    Outcome from_label(std::string const& label, Names const& group_names) {
      auto it = std::find(group_names.begin(), group_names.end(), label);
      return to_outcome(static_cast<GroupId>(std::distance(group_names.begin(), it)));
    }

    json label_value(types::Outcome value, Names const& names) {
      if (!names.empty()) {
        return to_label(to_group_id(value), names);
      }
      return value;
    }

    json label_groups(std::set<types::GroupId> const& groups, Names const& names) {
      if (names.empty()) {
        return groups;
      }

      Names named;
      named.reserve(groups.size());

      for (auto g : groups) {
        named.push_back(to_label(g, names));
      }

      return named;
    }


    json model_to_json(Model const& model, Names const& group_names) {
      invariant(model.training_spec != nullptr, "Model serialization requires a training spec.");

      JsonModelVisitor visitor(group_names);
      model.accept(visitor);
      visitor.result["config"] = model.training_spec->to_json();
      return visitor.result;
    }

    Outcome decode_leaf(json const& j, Names const& names) {
      return !names.empty() ? from_label(j.get<std::string>(), names) : j.get<Outcome>();
    }

    GroupId decode_group(json const& j, Names const& names) {
      return !names.empty() ? to_group_id(from_label(j.get<std::string>(), names)) : j.get<GroupId>();
    }

    TreeNode::Ptr node_from_json(json const& j, Names const& names) {
      if (j.contains("value")) {
        auto leaf        = TreeLeaf::make(decode_leaf(j["value"], names));
        leaf->degenerate = j.value("degenerate", false);
        return leaf;
      }

      auto const proj_vec = j["projector"].get<std::vector<Feature>>();

      pp::Projector projector = Eigen::Map<pp::Projector const>(proj_vec.data(), static_cast<int>(proj_vec.size()));

      Feature const cut            = j["cutpoint"].get<Feature>();
      Feature const pp_index_value = j.value("pp_index_value", Feature(0));

      std::set<GroupId> groups;

      if (j.contains("groups")) {
        for (auto const& g : j["groups"]) {
          groups.insert(decode_group(g, names));
        }
      }

      auto node = TreeBranch::make(
          projector, cut, node_from_json(j["lower"], names), node_from_json(j["upper"], names), groups, pp_index_value
      );

      node->degenerate = node->degenerate || j.value("degenerate", false);
      return node;
    }
  }

  json to_json(types::OutcomeVector const& y, types::Names const& names) {
    if (names.empty()) {
      return y; // Eigen ADL serializer → flat float array
    }

    types::Names labels;
    labels.reserve(static_cast<std::size_t>(y.size()));

    for (Eigen::Index i = 0; i < y.size(); ++i) {
      labels.push_back(names[static_cast<std::size_t>(to_group_id(y(i)))]);
    }

    return labels;
  }

  template<> json Export<Model::Ptr>::to_json() const {
    json result = serialization::to_json(*model, groups);

    json meta;
    meta["observations"]  = n_observations;
    meta["features"]      = n_features;
    meta["groups"]        = groups;
    meta["feature_names"] = feature_names;

    result["meta"]                = meta;
    result["training_metrics"]    = serialization::to_json(training_metrics, groups);
    result["oob_metrics"]         = serialization::to_json(oob_metrics, groups);
    result["variable_importance"] = serialization::to_json(variable_importance);

    return result;
  }

  void JsonNodeVisitor::visit(TreeBranch const& node) {
    JsonNodeVisitor lower_visitor(group_names);
    node.lower->accept(lower_visitor);

    JsonNodeVisitor upper_visitor(group_names);
    node.upper->accept(upper_visitor);

    result = json{
        {"projector", node.projector},
        {"cutpoint", node.cutpoint},
        {"pp_index_value", node.pp_index_value},
        {"lower", lower_visitor.result},
        {"upper", upper_visitor.result},
        {"degenerate", node.degenerate},
        {"groups", label_groups(node.groups, group_names)},
    };
  }

  void JsonNodeVisitor::visit(TreeLeaf const& node) {
    result = json{
        {"value", label_value(node.value, group_names)},
        {"degenerate", node.degenerate},
    };
  }

  void JsonModelVisitor::visit(Tree const& tree) {
    result = json{{"model_type", "tree"}, {"model", to_json(tree, group_names)}};
  }

  void JsonModelVisitor::visit(Forest const& forest) {
    result = json{{"model_type", "forest"}, {"model", to_json(forest, group_names)}};
  }

  json to_json(Model const& model) {
    return model_to_json(model, Names{});
  }

  json to_json(Model const& model, Names const& group_names) {
    if (group_names.empty()) {
      return to_json(model);
    }
    return model_to_json(model, group_names);
  }

  json to_json(TreeNode const& node) {
    JsonNodeVisitor visitor;
    node.accept(visitor);
    return visitor.result;
  }

  json to_json(Tree const& tree) {
    return json{
        {"root", to_json(*tree.root)},
        {"degenerate", tree.degenerate},
    };
  }

  json to_json(BaggedTree const& tree) {
    json result              = to_json(*tree.model);
    result["sample_indices"] = tree.sample_indices;
    return result;
  }

  json to_json(Forest const& forest) {
    std::vector<json> trees_json;
    trees_json.reserve(forest.trees.size());

    for (auto const& tree : forest.trees) {
      trees_json.push_back(to_json(*tree));
    }

    return json{
        {"trees", trees_json},
        {"degenerate", forest.degenerate},
    };
  }

  json to_json(ConfusionMatrix const& cm) {
    json j;

    std::vector<std::vector<int>> matrix_data;
    for (int i = 0; i < cm.values.rows(); ++i) {
      std::vector<int> row;
      row.reserve(static_cast<std::size_t>(cm.values.cols()));
      for (int col = 0; col < cm.values.cols(); ++col) {
        row.emplace_back(cm.values(i, col));
      }

      matrix_data.push_back(row);
    }

    j["matrix"] = matrix_data;

    std::vector<int> labels;
    labels.reserve(static_cast<std::size_t>(cm.label_index.size()));
    for (auto const& [label, idx] : cm.label_index) {
      labels.push_back(label);
    }

    j["labels"]       = labels;
    j["group_errors"] = cm.group_errors();

    return j;
  }

  json to_json(VariableImportance const& vi) {
    int const p = static_cast<int>(vi.projections.size());

    json j;
    j["scale"]       = vi.scale;
    j["projections"] = vi.projections;

    if (vi.weighted_projections.size() == p) {
      j["weighted_projections"] = vi.weighted_projections;
    }

    if (vi.permuted.size() == p) {
      j["permuted"] = vi.permuted;
    }

    return j;
  }

  json to_json(RegressionMetrics const& rm) {
    return json{
        {"mse", rm.mse},
        {"mae", rm.mae},
        {"r_squared", rm.r_squared},
    };
  }

  json to_json(ClassificationMetrics const& cm) {
    return json{
        {"confusion_matrix", to_json(cm.confusion_matrix)},
        {"error_rate", cm.error_rate()},
    };
  }

  json to_json(ClassificationMetrics const& cm, Names const& group_names) {
    if (group_names.empty()) {
      return to_json(cm);
    }
    return json{
        {"confusion_matrix", to_json(cm.confusion_matrix, group_names)},
        {"error_rate", cm.error_rate()},
    };
  }

  json to_json(RegressionMetrics const& rm, Names const& /*group_names*/) {
    return to_json(rm);
  }

  json to_json(Metrics const& metrics) {
    return std::visit([](auto const& m) { return to_json(m); }, metrics);
  }

  json to_json(Metrics const& metrics, Names const& group_names) {
    if (group_names.empty()) {
      return to_json(metrics);
    }
    return std::visit([&](auto const& m) { return to_json(m, group_names); }, metrics);
  }

  json to_json(std::optional<VariableImportance> const& vi) {
    return vi ? to_json(*vi) : json(nullptr);
  }

  json to_json(std::optional<Metrics> const& metrics) {
    return metrics ? to_json(*metrics) : json(nullptr);
  }

  json to_json(std::optional<Metrics> const& metrics, Names const& group_names) {
    if (group_names.empty()) {
      return to_json(metrics);
    }
    return metrics ? to_json(*metrics, group_names) : json(nullptr);
  }

  json to_json(TreeNode const& node, Names const& group_names) {
    if (group_names.empty()) {
      return to_json(node);
    }
    JsonNodeVisitor visitor(group_names);
    node.accept(visitor);
    return visitor.result;
  }

  json to_json(Tree const& tree, Names const& group_names) {
    if (group_names.empty()) {
      return to_json(tree);
    }
    return json{
        {"root", to_json(*tree.root, group_names)},
        {"degenerate", tree.degenerate},
    };
  }

  json to_json(BaggedTree const& tree, Names const& group_names) {
    if (group_names.empty()) {
      return to_json(tree);
    }
    json result              = to_json(*tree.model, group_names);
    result["sample_indices"] = tree.sample_indices;
    return result;
  }

  json to_json(Forest const& forest, Names const& group_names) {
    if (group_names.empty()) {
      return to_json(forest);
    }
    std::vector<json> trees_json;
    trees_json.reserve(forest.trees.size());

    for (auto const& tree : forest.trees) {
      trees_json.push_back(to_json(*tree, group_names));
    }

    return json{
        {"trees", trees_json},
        {"degenerate", forest.degenerate},
    };
  }

  json to_json(ConfusionMatrix const& cm, Names const& group_names) {
    if (group_names.empty()) {
      return to_json(cm);
    }
    json j;

    std::vector<std::vector<int>> matrix_data;
    for (int i = 0; i < cm.values.rows(); ++i) {
      std::vector<int> row;
      row.reserve(static_cast<std::size_t>(cm.values.cols()));
      for (int col = 0; col < cm.values.cols(); ++col) {
        row.emplace_back(cm.values(i, col));
      }

      matrix_data.push_back(row);
    }

    j["matrix"] = matrix_data;

    types::Names labels;
    labels.reserve(static_cast<std::size_t>(cm.label_index.size()));
    for (auto const& [label, idx] : cm.label_index) {
      labels.emplace_back(group_names[static_cast<std::size_t>(label)]);
    }

    j["labels"]       = labels;
    j["group_errors"] = cm.group_errors();

    return j;
  }

  json to_json(FeatureMatrix const& matrix) {
    return json(matrix);
  }

  // -----------------------------------------------------------------------
  // Deserialization
  // -----------------------------------------------------------------------

  template<> ConfusionMatrix from_json<ConfusionMatrix>(json const& j) {
    ConfusionMatrix cm;
    auto const& matrix_data = j["matrix"];
    int n                   = static_cast<int>(matrix_data.size());
    cm.values               = types::Matrix<int>::Zero(n, n);

    for (int i = 0; i < n; ++i) {
      for (int col = 0; col < n; ++col) {
        cm.values(i, col) = matrix_data[static_cast<std::size_t>(i)][static_cast<std::size_t>(col)].get<int>();
      }
    }

    auto const& labels = j["labels"];

    for (int i = 0; i < n; ++i) {
      if (labels[static_cast<std::size_t>(i)].is_number_integer()) {
        cm.label_index[labels[static_cast<std::size_t>(i)].get<int>()] = i;
      } else {
        cm.label_index[i] = i;
      }
    }

    return cm;
  }

  template<> VariableImportance from_json<VariableImportance>(json const& j) {
    VariableImportance vi;
    vi.scale       = j["scale"].get<types::FeatureVector>();
    vi.projections = j["projections"].get<types::FeatureVector>();

    if (j.contains("weighted_projections") && !j["weighted_projections"].empty()) {
      vi.weighted_projections = j["weighted_projections"].get<types::FeatureVector>();
    }

    if (j.contains("permuted") && !j["permuted"].empty()) {
      vi.permuted = j["permuted"].get<types::FeatureVector>();
    }

    return vi;
  }

  template<> RegressionMetrics from_json<RegressionMetrics>(json const& j) {
    RegressionMetrics rm;
    rm.mse       = j.at("mse").get<double>();
    rm.mae       = j.at("mae").get<double>();
    rm.r_squared = j.at("r_squared").get<double>();
    return rm;
  }

  template<> ClassificationMetrics from_json<ClassificationMetrics>(json const& j) {
    return ClassificationMetrics(serialization::from_json<ConfusionMatrix>(j.at("confusion_matrix")));
  }

  Metrics metrics_from_json(json const& j, types::Mode mode) {
    Metrics metrics;
    if (mode == types::Mode::Regression) {
      metrics = serialization::from_json<RegressionMetrics>(j);
    } else {
      metrics = serialization::from_json<ClassificationMetrics>(j);
    }
    return metrics;
  }

  template<> void Export<Model::Ptr>::compute_metrics(types::FeatureMatrix const& x, types::OutcomeVector const& y) {
    struct MetricsVisitor : Model::Visitor {
      FeatureMatrix const& x;
      OutcomeVector const& y;
      int seed;
      Export<Model::Ptr>& self;

      MetricsVisitor(FeatureMatrix const& x, OutcomeVector const& y, int seed, Export<Model::Ptr>& self)
          : x(x)
          , y(y)
          , seed(seed)
          , self(self) {}

      void visit(ClassificationTree const& tree) override {
        self.training_metrics    = ClassificationMetrics(tree.predict(x).cast<GroupId>(), y.cast<GroupId>());
        self.variable_importance = ppforest2::variable_importance(tree, x);
      }

      void visit(RegressionTree const& tree) override {
        self.training_metrics    = RegressionMetrics(tree.predict(x), y);
        self.variable_importance = ppforest2::variable_importance(tree, x);
      }

      void visit(ClassificationForest const& forest) override {
        self.training_metrics = ClassificationMetrics(forest.predict(x).cast<GroupId>(), y.cast<GroupId>());
        if (auto m = ppforest2::oob_metrics(forest, x, y)) {
          self.oob_metrics = *m;
        }
        self.variable_importance = ppforest2::variable_importance(forest, x, y, seed);
      }

      void visit(RegressionForest const& forest) override {
        self.training_metrics = RegressionMetrics(forest.predict(x), y);
        if (auto m = ppforest2::oob_metrics(forest, x, y)) {
          self.oob_metrics = *m;
        }
        self.variable_importance = ppforest2::variable_importance(forest, x, y, seed);
      }
    };

    MetricsVisitor visitor(x, y, model->training_spec->seed, *this);
    model->accept(visitor);
  }
}

// -------------------------------------------------------------------------
// adl_serializer implementations — Export<T>
// -------------------------------------------------------------------------
namespace nlohmann {
  using namespace ppforest2;
  using namespace ppforest2::serialization;
  using json = nlohmann::json;

  namespace {
    template<typename T> Export<Model::Ptr> as_model_export(Export<std::unique_ptr<T>>&& src) {
      return {
          std::shared_ptr<T>(src.model.release()),
          std::move(src.groups),
          std::move(src.spec),
          src.n_observations,
          src.n_features,
          std::move(src.feature_names),
      };
    }

    Export<Model::Ptr> read_model_export(json const& j) {
      std::string const model_type = j.value("model_type", "tree");
      if (model_type == "forest") {
        return as_model_export(j.get<Export<Forest::Ptr>>());
      }
      if (model_type == "tree") {
        return as_model_export(j.get<Export<Tree::Ptr>>());
      }
      throw std::invalid_argument("Invalid model type: " + model_type);
    }

    struct ExportHeader {
      TrainingSpec::Ptr spec;
      Names groups;
      json const& meta;
    };

    ExportHeader read_export_header(json const& j) {
      auto spec        = TrainingSpec::from_json(j.at("config"));
      auto const& meta = j.at("meta");
      auto groups      = meta.value("groups", Names{});
      return ExportHeader{std::move(spec), std::move(groups), meta};
    }

    template<typename T> Export<T> make_export(T model, ExportHeader&& header) {
      return {
          std::move(model),
          std::move(header.groups),
          std::move(header.spec),
          header.meta.value("observations", 0),
          header.meta.value("features", 0),
          header.meta.value("feature_names", types::Names{}),
      };
    }
  }

  namespace {
    /**
     * @brief Canonical group-id set derived from a meta-`groups` names array.
     *
     * `meta.groups` carries the training labels as strings; the ids that
     * the in-memory model uses are the ascending integer indices of those
     * names. Returns an empty set when @p group_names is empty (regression
     * exports — no group concept). Validation upstream
     * (`validate_forest_export`) already enforces that classification
     * exports always carry a non-empty `meta.groups`, so an empty result
     * here implies regression.
     */
    std::set<GroupId> group_ids_from_meta(Names const& group_names) {
      std::set<GroupId> ids;
      for (std::size_t i = 0; i < group_names.size(); ++i) {
        ids.insert(static_cast<GroupId>(i));
      }
      return ids;
    }

    /**
     * @brief Concrete-tree factory branched on mode.
     *
     * @p training_groups is the canonical training-label set passed
     * through to `ClassificationTree`'s constructor; ignored for
     * regression. This local helper exists so each call site doesn't
     * re-spell the mode branch.
     */
    Tree::Ptr make_tree(TrainingSpec::Ptr spec, TreeNode::Ptr root, std::set<GroupId> training_groups) {
      if (is_regression(spec)) {
        return std::make_unique<RegressionTree>(std::move(root), std::move(spec));
      }
      return std::make_unique<ClassificationTree>(std::move(root), std::move(spec), std::move(training_groups));
    }

    /** @brief Concrete-forest factory branched on mode. Symmetric with `make_tree`. */
    Forest::Ptr make_forest(TrainingSpec::Ptr spec, std::set<GroupId> training_groups) {
      if (is_regression(spec)) {
        return std::make_unique<RegressionForest>(std::move(spec));
      }
      return std::make_unique<ClassificationForest>(std::move(spec), std::move(training_groups));
    }
  }

  Export<Tree::Ptr> adl_serializer<Export<Tree::Ptr>>::from_json(json const& j) {
    // Run skeleton/config/meta checks up-front with path-annotated errors.
    // Everything below this point can assume well-formed structure.
    validate_tree_export(j);

    auto header          = read_export_header(j);
    auto training_groups = group_ids_from_meta(header.groups);
    auto root            = node_from_json(j.at("model")["root"], header.groups);
    Tree::Ptr tree       = make_tree(header.spec, std::move(root), std::move(training_groups));

    return make_export(std::move(tree), std::move(header));
  }

  Export<Forest::Ptr> adl_serializer<Export<Forest::Ptr>>::from_json(json const& j) {
    validate_forest_export(j);

    auto header    = read_export_header(j);
    auto const& mj = j.at("model");

    // Canonical group ids from `meta.groups` (validated non-empty for
    // classification, absent for regression). Threaded into both the
    // forest's `groups` field and every `ClassificationTree` it owns —
    // both are mode-specific properties of the model that must match the
    // training run, not be re-derived from any individual tree's reachable
    // node groups (a degenerate-retry tree could shrink that set).
    auto const training_groups = group_ids_from_meta(header.groups);

    if (is_classification(header.spec)) {
      invariant(
          !header.groups.empty(),
          "JSON deserialization: classification forest export must include non-empty `meta.groups` "
          "(should have been caught by validate_forest_export)"
      );
    }

    Forest::Ptr forest = make_forest(header.spec, training_groups);
    forest->degenerate = mj.value("degenerate", false);

    for (auto const& tree_json : mj.at("trees")) {
      auto sample_indices = tree_json.value("sample_indices", std::vector<int>{});
      auto root           = node_from_json(tree_json.at("root"), header.groups);

      forest->add_tree(
          BaggedTree::make(make_tree(header.spec, std::move(root), training_groups), std::move(sample_indices))
      );
    }

    return make_export(std::move(forest), std::move(header));
  }

  Export<Model::Ptr> adl_serializer<Export<Model::Ptr>>::from_json(json const& j) {
    using serialization::has_value;

    Export<Model::Ptr> result = read_model_export(j);

    types::Mode const mode = result.spec->mode;

    if (has_value(j, "variable_importance")) {
      result.variable_importance = serialization::from_json<VariableImportance>(j["variable_importance"]);
    }
    if (has_value(j, "training_metrics")) {
      result.training_metrics = serialization::metrics_from_json(j["training_metrics"], mode);
    }
    if (has_value(j, "oob_metrics")) {
      result.oob_metrics = serialization::metrics_from_json(j["oob_metrics"], mode);
    }
    return result;
  }
}

namespace ppforest2::serialization {
  std::ostream& operator<<(std::ostream& os, TreeNode const& node) {
    return os << to_json(node).dump(2, ' ', false);
  }

  std::ostream& operator<<(std::ostream& os, TreeBranch const& condition) {
    return os << to_json(static_cast<TreeNode const&>(condition)).dump(2, ' ', false);
  }

  std::ostream& operator<<(std::ostream& os, TreeLeaf const& response) {
    return os << to_json(static_cast<TreeNode const&>(response)).dump(2, ' ', false);
  }

  std::ostream& operator<<(std::ostream& os, Tree const& tree) {
    return os << to_json(tree).dump(2, ' ', false);
  }

  std::ostream& operator<<(std::ostream& os, Forest const& forest) {
    return os << to_json(forest).dump(2, ' ', false);
  }
}
