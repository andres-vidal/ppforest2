#include <gtest/gtest.h>

#include "models/ClassificationTree.hpp"
#include "models/RegressionTree.hpp"
#include "models/Tree.hpp"
#include "models/TreeLeaf.hpp"
#include "test/TestSpec.hpp"

using namespace ppforest2;
using namespace ppforest2::types;

namespace {
  struct CallLog {
    int tree                = 0;
    int classification_tree = 0;
    int regression_tree     = 0;
  };

  ClassificationTree make_classification_tree() {
    return ClassificationTree(TreeLeaf::make(0), test::classification_spec(), ClassificationTree::Groups{0});
  }

  RegressionTree make_regression_tree() {
    return RegressionTree(TreeLeaf::make(0.0F), test::regression_spec());
  }
}

TEST(TreeVisitor, BimodalVisitorReceivesBothSubclassesViaDefaultDelegation) {
  // A visitor that overrides only the bimodal visit(Tree) gets called
  // for both ClassificationTree and RegressionTree — the default
  // implementations of visit(ClassificationTree) / visit(RegressionTree)
  // delegate to visit(Tree).

  struct TestVisitor : Model::Visitor {
    CallLog log = CallLog{};
    void visit(Tree const& /*tree*/) override { ++log.tree; }
  };

  TestVisitor visitor;

  make_classification_tree().accept(visitor);
  make_regression_tree().accept(visitor);

  EXPECT_EQ(visitor.log.tree, 2);
  EXPECT_EQ(visitor.log.classification_tree, 0);
  EXPECT_EQ(visitor.log.regression_tree, 0);
}

TEST(TreeVisitor, ModeSpecificOverridesWinOverBimodalOverload) {
  // When mode-specific overrides exist, they fire instead of visit(Tree).
  struct TestVisitor : Model::Visitor {
    CallLog log = CallLog{};
    void visit(Tree const& /*tree*/) override { ++log.tree; }
    void visit(ClassificationTree const& /*tree*/) override { ++log.classification_tree; }
    void visit(RegressionTree const& /*tree*/) override { ++log.regression_tree; }
  };

  TestVisitor visitor;

  make_classification_tree().accept(visitor);
  make_regression_tree().accept(visitor);

  EXPECT_EQ(visitor.log.tree, 0);
  EXPECT_EQ(visitor.log.classification_tree, 1);
  EXPECT_EQ(visitor.log.regression_tree, 1);
}

TEST(TreeVisitor, PartialOverrideDelegatesUnhandledMode) {
  // Overriding only ClassificationTree leaves RegressionTree to delegate
  // to visit(Tree). Useful as a sanity check for callers that handle one
  // mode and treat the other as an error or no-op.

  struct TestVisitor : Model::Visitor {
    CallLog log = CallLog{};
    void visit(Tree const& /*tree*/) override { ++log.tree; }
    void visit(ClassificationTree const& /*tree*/) override { ++log.classification_tree; }
  };

  TestVisitor visitor;

  make_classification_tree().accept(visitor);
  make_regression_tree().accept(visitor);

  EXPECT_EQ(visitor.log.tree, 1);                // RegressionTree -> visit(Tree)
  EXPECT_EQ(visitor.log.classification_tree, 1); // ClassificationTree -> mode-specific
  EXPECT_EQ(visitor.log.regression_tree, 0);
}
