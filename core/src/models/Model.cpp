#include "models/Model.hpp"
#include "models/Tree.hpp"
#include "models/Forest.hpp"
#include "models/ClassificationTree.hpp"
#include "models/RegressionTree.hpp"
#include "models/ClassificationForest.hpp"
#include "models/RegressionForest.hpp"
#include "utils/UserError.hpp"


namespace ppforest2 {
  void Model::Visitor::visit(ClassificationTree const& tree) {
    visit(static_cast<Tree const&>(tree));
  }
  void Model::Visitor::visit(ClassificationForest const& forest) {
    visit(static_cast<Forest const&>(forest));
  }
  void Model::Visitor::visit(RegressionTree const& tree) {
    visit(static_cast<Tree const&>(tree));
  }
  void Model::Visitor::visit(RegressionForest const& forest) {
    visit(static_cast<Forest const&>(forest));
  }

  Model::Ptr Model::train(TrainingSpec const& spec, types::FeatureMatrix& x, types::OutcomeVector& y) {
    Model::Ptr model;

    if (spec.is_forest()) {
      model = std::shared_ptr<Forest>(Forest::train(spec, x, y).release());
    } else {
      model = std::shared_ptr<Tree>(Tree::train(spec, x, y).release());
    }

    return model;
  }

  void Model::check_train_inputs(types::FeatureMatrix const& x, types::OutcomeVector const& y) {
    user_error(y.size() > 0, "Training requires a non-empty response vector.");
    user_error(y.size() == x.rows(), "Response length does not match number of rows in the data.");
  }

  namespace {
    template<typename Input> struct Visitor : Model::Visitor {
      Input const& x;
      Input result;

      explicit Visitor(Input const& x)
          : x(x) {}

      void not_supported() { throw UserError("Vote proportions are not available for regression models."); }

      void visit(Tree const& /*tree*/) override { not_supported(); }
      void visit(Forest const& /*forest*/) override { not_supported(); }
      void visit(ClassificationTree const& tree) override { result = tree.predict(x, Proportions{}); }
      void visit(ClassificationForest const& forest) override { result = forest.predict(x, Proportions{}); }
    };
  }

  types::FeatureMatrix predict_proportions(Model const& model, types::FeatureMatrix const& x) {
    Visitor visitor(x);
    model.accept(visitor);
    return visitor.result;
  }

  types::FeatureVector predict_proportions(Model const& model, types::FeatureVector const& x) {
    Visitor visitor(x);
    model.accept(visitor);
    return visitor.result;
  }
}
