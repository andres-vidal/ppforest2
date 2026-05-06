#include <gtest/gtest.h>
#include <Eigen/Dense>

#include "models/Model.hpp"
#include "utils/Types.hpp"

using namespace ppforest2;
using namespace ppforest2::types;

namespace {
  // Minimal Model subclass. Implements only the single-row predict and
  // counts both `predict` and `accept` invocations so we can verify the
  // dispatch contract of `Model::predict(FeatureMatrix)` and the
  // `Model::Visitor`-based free functions (e.g. `predict_proportions`).
  class MockModel : public Model {
  public:
    mutable int single_row_calls = 0;
    mutable int accept_calls     = 0;

    Outcome predict(FeatureVector const& data) const override {
      ++single_row_calls;
      return data.sum();
    }

    using Model::predict;

    void accept(Visitor& /*visitor*/) const override { ++accept_calls; }
  };
}

TEST(ModelPredictMatrix, DispatchesToSingleRowForEachRow) {
  MockModel model;

  FeatureMatrix x(3, 2);
  x << 1.0F, 2.0F, 10.0F, 20.0F, 100.0F, 200.0F;

  OutcomeVector const out = model.predict(x);

  ASSERT_EQ(out.size(), 3);
  EXPECT_FLOAT_EQ(out(0), 3.0F);
  EXPECT_FLOAT_EQ(out(1), 30.0F);
  EXPECT_FLOAT_EQ(out(2), 300.0F);
  EXPECT_EQ(model.single_row_calls, 3);
}

TEST(PredictProportions, VisitsModel) {
  MockModel model;

  FeatureMatrix const x      = FeatureMatrix::Zero(2, 2);
  FeatureMatrix const result = predict_proportions(model, x);

  EXPECT_EQ(model.accept_calls, 1);
  EXPECT_EQ(result.rows(), 0);
  EXPECT_EQ(result.cols(), 0);
}


TEST(ModelPredictMatrix, EmptyMatrixReturnsEmptyVector) {
  MockModel model;

  FeatureMatrix const x(0, 4);
  OutcomeVector const out = model.predict(x);

  EXPECT_EQ(out.size(), 0);
  EXPECT_EQ(model.single_row_calls, 0);
}

TEST(ModelPredictMatrix, DispatchesViaBasePointer) {
  MockModel model;
  Model const& base = model;

  FeatureMatrix x(2, 2);
  x << 1.0F, 1.0F, 2.0F, 2.0F;

  OutcomeVector const out = base.predict(x);

  ASSERT_EQ(out.size(), 2);
  EXPECT_FLOAT_EQ(out(0), 2.0F);
  EXPECT_FLOAT_EQ(out(1), 4.0F);
  EXPECT_EQ(model.single_row_calls, 2);
}
