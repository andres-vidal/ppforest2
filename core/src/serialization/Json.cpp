#include "serialization/Json.hpp"
#include "models/Projector.hpp"
#include "models/TrainingSpecPDA.hpp"

#include <Eigen/Dense>
#include <algorithm>

using namespace ppforest2::types;

namespace ppforest2::serialization {
  std::vector<std::string> to_labels(
    const types::ResponseVector&    predictions,
    const std::vector<std::string>& group_names) {
    std::vector<std::string> labels;
    labels.reserve(static_cast<std::size_t>(predictions.size()));

    for (int i = 0; i < predictions.size(); ++i) {
      labels.push_back(group_names[static_cast<std::size_t>(predictions(i))]);
    }

    return labels;
  }

  json build_meta_json(
    int                             n_observations,
    int                             n_features,
    const std::vector<std::string>& group_names,
    const std::vector<std::string>& feature_names) {
    json meta;
    meta["observations"] = n_observations;
    meta["features"]     = n_features;

    if (!group_names.empty()) {
      meta["groups"] = group_names;
    }

    if (!feature_names.empty()) {
      meta["feature_names"] = feature_names;
    }

    return meta;
  }

  json build_model_json(
    const Model&                    model,
    const json&                     config,
    const std::vector<std::string>& group_names,
    const std::vector<std::string>& feature_names,
    int                             n_observations,
    int                             n_features) {
    json output = group_names.empty()
      ? to_json(model)
      : to_json(model, group_names);

    output["config"] = config;
    output["meta"]   = build_meta_json(n_observations, n_features, group_names, feature_names);

    return output;
  }

  void JsonNodeVisitor::visit(const TreeCondition& node) {
    JsonNodeVisitor lower_visitor;
    lower_visitor.group_names = group_names;
    node.lower->accept(lower_visitor);

    JsonNodeVisitor upper_visitor;
    upper_visitor.group_names = group_names;
    node.upper->accept(upper_visitor);

    result = json{
      { "projector",      node.projector },
      { "threshold",      node.threshold },
      { "pp_index_value", node.pp_index_value },
      { "lower",          lower_visitor.result },
      { "upper",          upper_visitor.result },
      { "degenerate",     node.degenerate },
    };

    if (group_names) {
      std::vector<std::string> named_groups;
      for (auto g : node.groups) {
        named_groups.push_back((*group_names)[static_cast<std::size_t>(g)]);
      }

      result["groups"] = named_groups;
    } else {
      result["groups"] = node.groups;
    }
  }

  void JsonNodeVisitor::visit(const TreeResponse& node) {
    if (group_names) {
      result = json{
        { "value",      (*group_names)[static_cast<std::size_t>(node.value)] },
        { "degenerate", node.degenerate },
      };
    } else {
      result = json{
        { "value",      node.value },
        { "degenerate", node.degenerate },
      };
    }
  }

  void JsonModelVisitor::visit(const Tree& tree) {
    result = group_names
      ? json{ { "model_type", "tree" }, { "model", to_json(tree, *group_names) } }
      : json{ { "model_type", "tree" }, { "model", to_json(tree) } };
  }

  void JsonModelVisitor::visit(const Forest& forest) {
    result = group_names
      ? json{ { "model_type", "forest" }, { "model", to_json(forest, *group_names) } }
      : json{ { "model_type", "forest" }, { "model", to_json(forest) } };
  }

  json to_json(const Model& model) {
    JsonModelVisitor visitor;
    model.accept(visitor);
    return visitor.result;
  }

  json to_json(const Model& model, const GroupNames& group_names) {
    JsonModelVisitor visitor;
    visitor.group_names = &group_names;
    model.accept(visitor);
    return visitor.result;
  }

  json to_json(const TreeNode& node) {
    JsonNodeVisitor visitor;
    node.accept(visitor);
    return visitor.result;
  }

  json to_json(const Tree& tree) {
    return json{
      { "root",       to_json(*tree.root) },
      { "degenerate", tree.degenerate },
    };
  }

  json to_json(const BootstrapTree& tree) {
    json result = to_json(static_cast<const Tree&>(tree));
    result["sample_indices"] = tree.sample_indices;
    return result;
  }

  json to_json(const Forest& forest) {
    std::vector<json> trees_json;
    trees_json.reserve(forest.trees.size());

    for (const auto& tree : forest.trees) {
      trees_json.push_back(to_json(*tree));
    }

    return json{
      { "trees",      trees_json },
      { "degenerate", forest.degenerate },
    };
  }

  json to_json(const stats::ConfusionMatrix& cm) {
    json j;

    std::vector<std::vector<int>> matrix_data;
    for (int i = 0; i < cm.values.rows(); ++i) {
      std::vector<int> row;
      for (int col = 0; col < cm.values.cols(); ++col) {
        row.push_back(cm.values(i, col));
      }

      matrix_data.push_back(row);
    }

    j["matrix"] = matrix_data;

    std::vector<int> labels;
    for (const auto& [label, idx] : cm.label_index) {
      labels.push_back(label);
    }

    j["labels"] = labels;

    auto ce = cm.group_errors();
    std::vector<float> ce_vec(ce.data(), ce.data() + ce.size());
    j["group_errors"] = ce_vec;

    return j;
  }

  json to_json(const VariableImportance& vi) {
    const int p = static_cast<int>(vi.projections.size());

    std::vector<float> scale_vec(vi.scale.data(), vi.scale.data() + p);
    std::vector<float> proj_vec(vi.projections.data(), vi.projections.data() + p);

    json j;
    j["scale"]       = scale_vec;
    j["projections"] = proj_vec;

    if (vi.weighted_projections.size() == p) {
      std::vector<float> wp_vec(vi.weighted_projections.data(), vi.weighted_projections.data() + p);
      j["weighted_projections"] = wp_vec;
    }

    if (vi.permuted.size() == p) {
      std::vector<float> perm_vec(vi.permuted.data(), vi.permuted.data() + p);
      j["permuted"] = perm_vec;
    }

    return j;
  }

  json to_json(const TreeNode& node, const GroupNames& group_names) {
    JsonNodeVisitor visitor;
    visitor.group_names = &group_names;
    node.accept(visitor);
    return visitor.result;
  }

  json to_json(const Tree& tree, const GroupNames& group_names) {
    return json{
      { "root",       to_json(*tree.root, group_names) },
      { "degenerate", tree.degenerate },
    };
  }

  json to_json(const BootstrapTree& tree, const GroupNames& group_names) {
    json result = to_json(static_cast<const Tree&>(tree), group_names);
    result["sample_indices"] = tree.sample_indices;
    return result;
  }

  json to_json(const Forest& forest, const GroupNames& group_names) {
    std::vector<json> trees_json;
    trees_json.reserve(forest.trees.size());

    for (const auto& tree : forest.trees) {
      trees_json.push_back(to_json(*tree, group_names));
    }

    return json{
      { "trees",      trees_json },
      { "degenerate", forest.degenerate },
    };
  }

  json to_json(const stats::ConfusionMatrix& cm, const GroupNames& group_names) {
    json j;

    std::vector<std::vector<int>> matrix_data;
    for (int i = 0; i < cm.values.rows(); ++i) {
      std::vector<int> row;
      for (int col = 0; col < cm.values.cols(); ++col) {
        row.push_back(cm.values(i, col));
      }

      matrix_data.push_back(row);
    }

    j["matrix"] = matrix_data;

    std::vector<std::string> labels;
    for (const auto& [label, idx] : cm.label_index) {
      labels.push_back(group_names[static_cast<std::size_t>(label)]);
    }

    j["labels"] = labels;

    auto ce = cm.group_errors();
    std::vector<float> ce_vec(ce.data(), ce.data() + ce.size());
    j["group_errors"] = ce_vec;

    return j;
  }

  json to_json(const FeatureMatrix& matrix) {
    std::vector<std::vector<Feature>> rows;
    rows.reserve(static_cast<std::size_t>(matrix.rows()));

    for (int i = 0; i < matrix.rows(); ++i) {
      std::vector<Feature> row(static_cast<std::size_t>(matrix.cols()));

      for (int j = 0; j < matrix.cols(); ++j) {
        row[static_cast<std::size_t>(j)] = matrix(i, j);
      }

      rows.push_back(std::move(row));
    }

    return json(rows);
  }

  static bool has_string_labels(const json& model_json) {
    const json *node = nullptr;

    if (model_json.contains("root")) {
      node = &model_json["root"];
    } else if (model_json.contains("trees") && !model_json["trees"].empty()) {
      node = &model_json["trees"][0]["root"];
    }

    if (!node) return false;

    while (node->contains("lower"))
      node = &(*node)["lower"];

    return node->contains("value") && (*node)["value"].is_string();
  }

  Model::Ptr model_from_json(const json& j) {
    std::string model_type = j.value("model_type", "tree");
    const auto& model_json = j["model"];

    bool labeled = has_string_labels(model_json)
      && j.contains("meta") && j["meta"].contains("groups");

    if (model_type == "forest") {
      return labeled
        ? std::make_unique<Forest>(forest_from_json(model_json, j["meta"]["groups"].get<GroupNames>()))
        : std::make_unique<Forest>(forest_from_json(model_json));
    } else if (model_type == "tree") {
      return labeled
        ? std::make_unique<Tree>(tree_from_json(model_json, j["meta"]["groups"].get<GroupNames>()))
        : std::make_unique<Tree>(tree_from_json(model_json));
    } else {
      throw std::invalid_argument("Invalid model type: " + model_type);
    }
  }

  TreeNode::Ptr node_from_json(const json& j) {
    if (j.contains("value")) {
      auto leaf = TreeResponse::make(j["value"].get<Response>());
      leaf->degenerate = j.value("degenerate", false);
      return leaf;
    }

    const auto proj_vec = j["projector"].get<std::vector<Feature>>();

    pp::Projector projector = Eigen::Map<const pp::Projector>(
      proj_vec.data(),
      static_cast<int>(proj_vec.size()));

    const Feature threshold      = j["threshold"].get<Feature>();
    const Feature pp_index_value = j.value("pp_index_value", Feature(0));

    std::set<Response> groups;

    if (j.contains("groups")) {
      groups = j["groups"].get<std::set<Response>>();
    }

    auto lower = node_from_json(j["lower"]);
    auto upper = node_from_json(j["upper"]);

    auto node = TreeCondition::make(
      projector, threshold,
      std::move(lower), std::move(upper),
      nullptr, groups, pp_index_value);
    node->degenerate = node->degenerate || j.value("degenerate", false);
    return node;
  }

  Tree tree_from_json(const json& j) {
    return Tree(node_from_json(j["root"]));
  }

  Forest forest_from_json(const json& j) {
    Forest forest;
    forest.degenerate = j.value("degenerate", false);

    for (const auto& tree_json : j.at("trees")) {
      auto sample_indices = tree_json.contains("sample_indices")
        ? tree_json["sample_indices"].get<std::vector<int>>()
        : std::vector<int>{};

      forest.add_tree(std::make_unique<BootstrapTree>(
          node_from_json(tree_json.at("root")),
          TrainingSpecPDA::make(0.5),
          std::move(sample_indices)));
    }

    return forest;
  }

  static Response label_to_code(const std::string& label, const GroupNames& group_names) {
    auto it = std::find(group_names.begin(), group_names.end(), label);
    return static_cast<Response>(std::distance(group_names.begin(), it));
  }

  TreeNode::Ptr node_from_json(const json& j, const GroupNames& group_names) {
    if (j.contains("value")) {
      auto leaf = TreeResponse::make(label_to_code(j["value"].get<std::string>(), group_names));
      leaf->degenerate = j.value("degenerate", false);
      return leaf;
    }

    const auto proj_vec = j["projector"].get<std::vector<Feature>>();

    pp::Projector projector = Eigen::Map<const pp::Projector>(
      proj_vec.data(),
      static_cast<int>(proj_vec.size()));

    const Feature threshold      = j["threshold"].get<Feature>();
    const Feature pp_index_value = j.value("pp_index_value", Feature(0));

    std::set<Response> groups;

    if (j.contains("groups")) {
      for (const auto& name : j["groups"]) {
        groups.insert(label_to_code(name.get<std::string>(), group_names));
      }
    }

    auto lower = node_from_json(j["lower"], group_names);
    auto upper = node_from_json(j["upper"], group_names);

    auto node = TreeCondition::make(
      projector, threshold,
      std::move(lower), std::move(upper),
      nullptr, groups, pp_index_value);
    node->degenerate = node->degenerate || j.value("degenerate", false);
    return node;
  }

  Tree tree_from_json(const json& j, const GroupNames& group_names) {
    return Tree(node_from_json(j["root"], group_names));
  }

  Forest forest_from_json(const json& j, const GroupNames& group_names) {
    Forest forest;
    forest.degenerate = j.value("degenerate", false);

    for (const auto& tree_json : j.at("trees")) {
      auto sample_indices = tree_json.contains("sample_indices")
        ? tree_json["sample_indices"].get<std::vector<int>>()
        : std::vector<int>{};

      forest.add_tree(std::make_unique<BootstrapTree>(
          node_from_json(tree_json.at("root"), group_names),
          TrainingSpecPDA::make(0.5),
          std::move(sample_indices)));
    }

    return forest;
  }

  stats::ConfusionMatrix confusion_matrix_from_json(const json& j) {
    stats::ConfusionMatrix cm;

    auto matrix_data = j["matrix"];
    int n            = static_cast<int>(matrix_data.size());
    cm.values = types::Matrix<int>::Zero(n, n);

    for (int i = 0; i < n; ++i) {
      for (int col = 0; col < n; ++col) {
        cm.values(i, col) = matrix_data[static_cast<std::size_t>(i)][static_cast<std::size_t>(col)].get<int>();
      }
    }

    const auto& labels = j["labels"];

    for (int i = 0; i < n; ++i) {
      if (labels[static_cast<std::size_t>(i)].is_number_integer()) {
        cm.label_index[labels[static_cast<std::size_t>(i)].get<int>()] = i;
      } else {
        cm.label_index[i] = i;
      }
    }

    return cm;
  }

  VariableImportance variable_importance_from_json(const json& j) {
    VariableImportance vi;

    auto scale_vec = j["scale"].get<std::vector<float>>();
    auto proj_vec  = j["projections"].get<std::vector<float>>();
    int p          = static_cast<int>(proj_vec.size());

    vi.scale       = Eigen::Map<const types::FeatureVector>(scale_vec.data(), p);
    vi.projections = Eigen::Map<const types::FeatureVector>(proj_vec.data(), p);

    if (j.contains("weighted_projections") && !j["weighted_projections"].empty()) {
      auto wp_vec = j["weighted_projections"].get<std::vector<float>>();
      vi.weighted_projections = Eigen::Map<const types::FeatureVector>(wp_vec.data(), p);
    }

    if (j.contains("permuted") && !j["permuted"].empty()) {
      auto perm_vec = j["permuted"].get<std::vector<float>>();
      vi.permuted = Eigen::Map<const types::FeatureVector>(perm_vec.data(), p);
    }

    return vi;
  }

  std::ostream& operator<<(std::ostream& os, const TreeNode& node) {
    return os << to_json(node).dump(2, ' ', false);
  }

  std::ostream& operator<<(std::ostream& os, const TreeCondition& condition) {
    return os << to_json(static_cast<const TreeNode&>(condition)).dump(2, ' ', false);
  }

  std::ostream& operator<<(std::ostream& os, const TreeResponse& response) {
    return os << to_json(static_cast<const TreeNode&>(response)).dump(2, ' ', false);
  }

  std::ostream& operator<<(std::ostream& os, const Tree& tree) {
    return os << to_json(tree).dump(2, ' ', false);
  }

  std::ostream& operator<<(std::ostream& os, const Forest& forest) {
    return os << to_json(forest).dump(2, ' ', false);
  }
}
