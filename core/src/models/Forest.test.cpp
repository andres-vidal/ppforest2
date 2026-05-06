#include <gtest/gtest.h>

#include "models/ClassificationForest.hpp"
#include "models/Forest.hpp"
#include "models/RegressionForest.hpp"
#include "test/TestSpec.hpp"

using namespace ppforest2;

namespace {
  struct CallLog {
    int forest                = 0;
    int classification_forest = 0;
    int regression_forest     = 0;
  };
}

TEST(ForestVisitor, BimodalVisitorReceivesBothSubclassesViaDefaultDelegation) {
  // A visitor that overrides only the bimodal visit(Forest) gets called
  // for both ClassificationForest and RegressionForest — the default
  // implementations of visit(ClassificationForest) / visit(RegressionForest)
  // delegate to visit(Forest).
  struct TestVisitor : Model::Visitor {
    CallLog log = CallLog{};
    void visit(Forest const& /*forest*/) override { ++log.forest; }
  };

  TestVisitor visitor;

  ClassificationForest(test::classification_spec(), ClassificationForest::Groups{0, 1}).accept(visitor);
  RegressionForest(test::regression_spec()).accept(visitor);

  EXPECT_EQ(visitor.log.forest, 2);
  EXPECT_EQ(visitor.log.classification_forest, 0);
  EXPECT_EQ(visitor.log.regression_forest, 0);
}

TEST(ForestVisitor, ModeSpecificOverridesWinOverBimodalOverload) {
  // When mode-specific overrides exist, they fire instead of visit(Forest).
  struct TestVisitor : Model::Visitor {
    CallLog log = CallLog{};
    void visit(Forest const& /*forest*/) override { ++log.forest; }
    void visit(ClassificationForest const& /*forest*/) override { ++log.classification_forest; }
    void visit(RegressionForest const& /*forest*/) override { ++log.regression_forest; }
  };

  TestVisitor visitor;

  ClassificationForest(test::classification_spec(), ClassificationForest::Groups{0, 1}).accept(visitor);
  RegressionForest(test::regression_spec()).accept(visitor);

  EXPECT_EQ(visitor.log.forest, 0);
  EXPECT_EQ(visitor.log.classification_forest, 1);
  EXPECT_EQ(visitor.log.regression_forest, 1);
}

TEST(ForestVisitor, PartialOverrideDelegatesUnhandledMode) {
  // Overriding only ClassificationForest leaves RegressionForest to
  // delegate to visit(Forest). This is the pattern used by R bindings
  // and the CLI to route mode mismatches to a UserError.
  struct TestVisitor : Model::Visitor {
    CallLog log = CallLog{};
    void visit(Forest const& /*forest*/) override { ++log.forest; }
    void visit(ClassificationForest const& /*forest*/) override { ++log.classification_forest; }
  };

  TestVisitor visitor;

  ClassificationForest(test::classification_spec(), ClassificationForest::Groups{0, 1}).accept(visitor);
  RegressionForest(test::regression_spec()).accept(visitor);

  EXPECT_EQ(visitor.log.forest, 1);                // RegressionForest -> visit(Forest)
  EXPECT_EQ(visitor.log.classification_forest, 1); // ClassificationForest -> mode-specific
  EXPECT_EQ(visitor.log.regression_forest, 0);
}
