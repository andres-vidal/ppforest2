#include <gtest/gtest.h>
#include <Eigen/Dense>

#include "models/Bagged.hpp"
#include "models/Model.hpp"
#include "utils/Macros.hpp"
#include "utils/Types.hpp"

#include <functional>
#include <memory>
#include <utility>
#include <vector>

using namespace ppforest2;
using namespace ppforest2::types;

namespace {
  struct MockModel : Model {
    using PredictFn = std::function<Outcome(FeatureVector const&)>;
    using Model::predict;

    PredictFn on_predict;

    explicit MockModel(PredictFn fn = {})
        : on_predict(std::move(fn)) {}

    Outcome predict(FeatureVector const& x) const override { return on_predict ? on_predict(x) : Outcome{}; }

    void accept(Model::Visitor& /*v*/) const override {}

    bool operator==(MockModel const& other) const { return this == &other; }
  };

  using MockBag = Bagged<MockModel>;

  // Tests that don't care what the model does — just that the OOB
  // bookkeeping is right — get a default-constructed mock.
  std::unique_ptr<MockModel> trivial_model() {
    return std::make_unique<MockModel>();
  }
}

// ---------------------------------------------------------------------------
// oob_indices
// ---------------------------------------------------------------------------

TEST(BaggedOOBIndices, ComplementOfSampleIndices) {
  // sample_indices = {0, 1, 2}, n_total = 5  =>  OOB = {3, 4}
  MockBag const bag(trivial_model(), std::vector<int>{0, 1, 2});

  auto oob = bag.oob_indices(5);

  ASSERT_EQ(oob.size(), 2U);
  ASSERT_EQ(oob[0], 3);
  ASSERT_EQ(oob[1], 4);
}

TEST(BaggedOOBIndices, EmptyWhenAllInBag) {
  MockBag const bag(trivial_model(), std::vector<int>{0, 1, 2, 3});

  auto oob = bag.oob_indices(4);
  ASSERT_TRUE(oob.empty());
}

TEST(BaggedOOBIndices, AllOOBWhenNoneInBag) {
  MockBag const bag(trivial_model(), std::vector<int>{});

  auto oob = bag.oob_indices(3);

  ASSERT_EQ(oob.size(), 3U);
  ASSERT_EQ(oob[0], 0);
  ASSERT_EQ(oob[1], 1);
  ASSERT_EQ(oob[2], 2);
}

TEST(BaggedOOBIndices, DuplicatesInSampleCountedOnce) {
  // With duplicates {0, 0, 1} the in-bag set is {0, 1}, OOB = {2, 3}
  MockBag const bag(trivial_model(), std::vector<int>{0, 0, 1});

  auto oob = bag.oob_indices(4);

  ASSERT_EQ(oob.size(), 2U);
  ASSERT_EQ(oob[0], 2);
  ASSERT_EQ(oob[1], 3);
}

// ---------------------------------------------------------------------------
// predict_oob
// ---------------------------------------------------------------------------

TEST(BaggedPredictOOB, MatchesRowwisePredict) {
  // Mock that thresholds x[0] at 5.0 — a stand-in for any model whose
  // single-row predictions vary over the input.
  MockBag const bag(
      std::make_unique<MockModel>([](FeatureVector const& v) -> Outcome { return v(0) < 5.0F ? 0 : 1; }),
      std::vector<int>{0, 1, 4, 5}
  );

  FeatureMatrix x = MAT(Feature, rows(6), 0.0F, 0.5F, 0.1F, 0.3F, 0.2F, 0.7F, 9.8F, 0.4F, 9.9F, 0.6F, 9.7F, 0.2F);

  std::vector<int> const oob_idx = {2, 3};
  OutcomeVector preds            = bag.predict_oob(x, oob_idx);

  ASSERT_EQ(preds.size(), 2);
  EXPECT_EQ(preds(0), 0) << "Row 2 has x[0]=0.2 < 5";
  EXPECT_EQ(preds(1), 1) << "Row 3 has x[0]=9.8 > 5";
}

TEST(BaggedPredictOOB, EmptyIndicesReturnsEmptyVector) {
  MockBag const bag(trivial_model(), std::vector<int>{0, 1});

  FeatureMatrix x(4, 2);
  x << 0, 0, 1, 1, 9, 9, 8, 8;

  OutcomeVector const preds = bag.predict_oob(x, std::vector<int>{});

  ASSERT_EQ(preds.size(), 0);
}
