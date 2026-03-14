#include "models/Tree.hpp"

#include "models/TreeCondition.hpp"
#include "models/TreeResponse.hpp"
#include "models/ModelVisitor.hpp"
#include "models/BootstrapTree.hpp"
#include "utils/Invariant.hpp"
#include "utils/Map.hpp"
#include "models/TrainingSpecGLDA.hpp"

#include <stack>
#include <Eigen/Dense>

using namespace ppforest2::pp;
using namespace ppforest2::dr;
using namespace ppforest2::sr;
using namespace ppforest2::stats;
using namespace ppforest2::types;
using namespace ppforest2::utils;

namespace ppforest2 {
  std::map<Response, int> binary_regroup(
    const FeatureMatrix &  x,
    const GroupPartition & group_spec
    ) {
    std::vector<std::tuple<Response, Feature>> means;

    invariant(x.cols() == 1, "Binary regrouping requires a unidimensional data");

    for (const Response group : group_spec.groups) {
      Feature group_mean = group_spec.group(x, group).mean();

      means.push_back({ group, group_mean });
    }

    std::sort(means.begin(), means.end(), [](const auto &a, const auto &b) {
        return std::get<1>(a) < std::get<1>(b);
      });

    Feature edge_gap    = -1;
    Response edge_group = -1;

    for (size_t i = 0; i + 1 < means.size(); i++) {
      Feature gap = std::get<1>(means[i + 1]) - std::get<1>(means[i]);

      if (gap > edge_gap) {
        edge_gap   = gap;
        edge_group = std::get<0>(means[i + 1]);
      }
    }

    if (edge_group == -1) {
      edge_group = std::get<0>(means.front());
    }

    std::map<Response, int > binary_mapping;

    bool edge_found = false;
    for (const auto&[group, mean] : means) {
      edge_found = edge_found || group == edge_group;

      binary_mapping[group] = edge_found ? 1 : 0;
    }

    return binary_mapping;
  }

  TreeCondition::Ptr binary_step(
    const TrainingSpec &   training_spec,
    const FeatureMatrix &  x,
    const GroupPartition & group_spec,
    const DRSpec&          dr) {
    Response group_1 = *group_spec.groups.begin();
    Response group_2 = *std::next(group_spec.groups.begin());

    const PPStrategy &pp_strategy = *(training_spec.pp_strategy);

    auto reduced_x = x(Eigen::placeholders::all, dr.selected_cols);

    auto [reduced_projector, pp_index_value] = pp_strategy.optimize(reduced_x, group_spec);
    Projector projector = dr.expand(reduced_projector);

    auto data_group_1 = group_spec.group(x, group_1);
    auto data_group_2 = group_spec.group(x, group_2);

    const SRStrategy &sr_strategy = *(training_spec.sr_strategy);
    Feature threshold             = sr_strategy.threshold(data_group_1, data_group_2, projector);

    Feature projected_mean_1 = data_group_1.colwise().mean().dot(projector);
    Feature projected_mean_2 = data_group_2.colwise().mean().dot(projector);

    Response lower_group, upper_group;

    if (projected_mean_1 < projected_mean_2) {
      lower_group = group_1;
      upper_group = group_2;
    } else {
      lower_group = group_2;
      upper_group = group_1;
    }

    TreeResponse::Ptr lower_response = TreeResponse::make(lower_group);
    TreeResponse::Ptr upper_response = TreeResponse::make(upper_group);

    auto condition = TreeCondition::make(
      projector,
      threshold,
      std::move(lower_response),
      std::move(upper_response),
      training_spec.clone(),
      group_spec.groups,
      pp_index_value);

    return condition;
  }

  GroupPartition binary_split(
    const TrainingSpec &   training_spec,
    const FeatureMatrix &  x,
    const GroupPartition & group_spec,
    const DRSpec&          dr
    ) {
    const PPStrategy &pp_strategy = *(training_spec.pp_strategy);

    FeatureMatrix reduced_x = x(Eigen::placeholders::all, dr.selected_cols);

    Projector projector = dr.expand(pp_strategy(reduced_x, group_spec));

    FeatureMatrix projected_x = x * projector;

    std::map<Response, int> binary_mapping = binary_regroup(projected_x, group_spec);


    return group_spec.remap(binary_mapping);
  }

  struct Step {
    GroupPartition y;
    TreeNode::Ptr *node;

    bool pop               = false;
    TreeNode::Ptr upper    = nullptr;
    TreeNode::Ptr lower    = nullptr;
    Threshold threshold    = 0;
    Feature pp_index_value = 0;
    Projector projector;

    Step(
      const GroupPartition& y,
      TreeNode::Ptr         *node,
      const int             cols
      ) :
      y(y),
      node(node),
      projector(Projector::Zero(cols)) {
    }
  };

  TreeNode::Ptr build_root(
    const TrainingSpec &   training_spec,
    const FeatureMatrix &  x,
    const GroupPartition & y,
    stats::RNG &           rng
    ) {
    const PPStrategy &pp_strategy = *(training_spec.pp_strategy);
    const DRStrategy &dr_strategy = *(training_spec.dr_strategy);

    std::stack<Step> stack;

    TreeNode::Ptr root;

    stack.emplace(y, &root, x.cols());

    while (!stack.empty()) {
      Step& step = stack.top();

      if (step.pop) {
        *step.node = TreeCondition::make(
          step.projector,
          step.threshold,
          std::move(step.lower),
          std::move(step.upper),
          training_spec.clone(),
          step.y.groups,
          step.pp_index_value);

        stack.pop();
        continue;
      }

      if (step.y.groups.size() == 1) {
        *step.node = TreeResponse::make(*step.y.groups.begin());
        stack.pop();
        continue;
      }

      DRSpec dr = dr_strategy(x, step.y, rng);

      if (step.y.groups.size() == 2) {
        *step.node = binary_step(training_spec, x, step.y, dr);
        stack.pop();
        continue;
      }

      auto split     = binary_split(training_spec, x, step.y, dr);
      auto temp_node = binary_step(training_spec, x, split, dr);

      Response binary_lower_group = temp_node->lower->response();
      Response binary_upper_group = temp_node->upper->response();

      step.projector      = temp_node->projector;
      step.threshold      = temp_node->threshold;
      step.pp_index_value = temp_node->pp_index_value;

      auto lower_y = split.subset(split.subgroups.at(binary_lower_group));
      auto upper_y = split.subset(split.subgroups.at(binary_upper_group));

      stack.emplace(lower_y, &step.lower, x.cols());
      stack.emplace(upper_y, &step.upper, x.cols());

      step.pop = true;
    }

    return root;
  }

  Tree Tree::train(
    TrainingSpec const&   training_spec,
    const FeatureMatrix&  x,
    const ResponseVector& y,
    stats::RNG &          rng) {
    GroupPartition group_spec(y);

    return Tree::train(training_spec, x, group_spec, rng);
  }

  Tree Tree::train(
    TrainingSpec const &  training_spec,
    const FeatureMatrix&  x,
    GroupPartition const& group_spec,
    stats::RNG &          rng) {
    TreeNode::Ptr root_ptr = build_root(training_spec, x, group_spec, rng);

    Tree tree(
      std::move(root_ptr),
      training_spec.clone());

    return tree;
  }

  Tree::Tree(TreeNode::Ptr root) :
    root(std::move(root)),
    training_spec(TrainingSpecGLDA::make(0.5)) {
  }

  Tree::Tree(TreeNode::Ptr root, TrainingSpec::Ptr training_spec) :
    root(std::move(root)),
    training_spec(std::move(training_spec)) {
  }

  Response Tree::predict(const FeatureVector& data) const {
    return root->predict(data);
  }

  ResponseVector Tree::predict(const FeatureMatrix& data) const {
    ResponseVector predictions(data.rows());

    for (int i = 0; i < data.rows(); i++) {
      predictions(i) = predict((FeatureVector)data.row(i));
    }

    return predictions;
  }

  bool Tree::operator==(const Tree& other) const {
    return *root == *other.root;
  }

  bool Tree::operator!=(const Tree& other) const {
    return !(*this == other);
  }

  void Tree::accept(ModelVisitor& visitor) const {
    visitor.visit(*this);
  }
}
