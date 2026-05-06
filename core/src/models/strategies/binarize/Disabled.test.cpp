#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "models/strategies/binarize/Disabled.hpp"
#include "models/strategies/binarize/Binarization.hpp"
#include "test/NodeContextFixture.hpp"
#include "utils/Macros.hpp"

using namespace ppforest2;
using namespace ppforest2::binarize;
using namespace ppforest2::stats;
using namespace ppforest2::test;
using namespace ppforest2::types;
using json = nlohmann::json;

TEST(DisabledBinarize, FromJsonValid) {
  json const j = {{"name", "disabled"}};

  auto strategy = Disabled::from_json(j);

  ASSERT_NE(strategy, nullptr);
}

TEST(DisabledBinarize, FromJsonRoundTrip) {
  json const j = {{"name", "disabled"}};

  auto strategy = Disabled::from_json(j);

  EXPECT_EQ(strategy->to_json(), j);
}

TEST(DisabledBinarize, FromJsonUnknownParam) {
  json const j = {{"name", "disabled"}, {"extra", true}};
  EXPECT_THROW(Disabled::from_json(j), std::runtime_error);
}

TEST(DisabledBinarize, RegistryLookup) {
  json const j = {{"name", "disabled"}};

  auto strategy = Binarization::from_json(j);

  ASSERT_NE(strategy, nullptr);
  EXPECT_EQ(strategy->to_json(), j);
}

TEST(DisabledBinarize, SupportsBothModes) {
  Disabled const d;
  auto const modes = d.supported_modes();
  EXPECT_EQ(modes.count(types::Mode::Classification), 1U);
  EXPECT_EQ(modes.count(types::Mode::Regression), 1U);
}

TEST(DisabledBinarize, DisplayName) {
  Disabled const d;
  EXPECT_EQ(d.display_name(), "Disabled");
}

TEST(DisabledBinarize, RegroupFiresInvariant) {
  // Disabled::regroup unconditionally raises an invariant — it is a
  // placeholder that must never actually be invoked in correct configs.
  NodeContextFixture f(MAT(Feature, rows(2), 1.0, 2.0, 3.0, 4.0), VEC(GroupId, 0, 1));

  Disabled const d;
  EXPECT_THROW(d.regroup(f.ctx, f.rng), std::exception);
}
