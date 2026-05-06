/**
 * @file TrainingSpec.test.cpp
 * @brief Unit tests for TrainingSpec composition, serialization, and defaults.
 */
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "models/TrainingSpec.hpp"
#include "utils/UserError.hpp"

using namespace ppforest2;
using json = nlohmann::json;

// ---------------------------------------------------------------------------
// is_forest()
// ---------------------------------------------------------------------------

TEST(TrainingSpec, IsForestTrue) {
  auto spec = TrainingSpec::builder(types::Mode::Classification)
                  .size(5)
                  .threads(2)
                  .pp(pp::pda(0.3F))
                  .vars(vars::uniform(2))
                  .make();
  EXPECT_TRUE(spec->is_forest());
}

TEST(TrainingSpec, IsForestFalse) {
  auto spec =
      TrainingSpec::builder(types::Mode::Classification).threads(2).pp(pp::pda(0.3F)).vars(vars::uniform(2)).make();
  EXPECT_FALSE(spec->is_forest());
}

// ---------------------------------------------------------------------------
// resolve_threads()
// ---------------------------------------------------------------------------

TEST(TrainingSpec, ResolveThreadsExplicit) {
  auto spec = TrainingSpec::builder(types::Mode::Classification)
                  .size(5)
                  .threads(4)
                  .pp(pp::pda(0.3F))
                  .vars(vars::uniform(2))
                  .make();
  EXPECT_EQ(spec->resolve_threads(), 4);
}

TEST(TrainingSpec, ResolveThreadsDefault) {
  auto spec =
      TrainingSpec::builder(types::Mode::Classification).size(5).pp(pp::pda(0.3F)).vars(vars::uniform(2)).make();
  EXPECT_GT(spec->resolve_threads(), 0);
}

// ---------------------------------------------------------------------------
// to_json / from_json round-trip
// ---------------------------------------------------------------------------

TEST(TrainingSpec, ToJsonRoundTrip) {
  auto spec = TrainingSpec::builder(types::Mode::Classification)
                  .size(5)
                  .threads(2)
                  .max_retries(5)
                  .pp(pp::pda(0.3F))
                  .vars(vars::uniform(2))
                  .make();

  auto j        = spec->to_json();
  auto restored = TrainingSpec::from_json(j);

  EXPECT_EQ(restored->size, 5);
  EXPECT_EQ(restored->seed, 0);
  EXPECT_EQ(restored->threads, 2);
  EXPECT_EQ(restored->max_retries, 5);

  EXPECT_EQ(j, restored->to_json());
}

TEST(TrainingSpec, ToJsonRoundTripSingleTree) {
  auto spec = TrainingSpec::builder(types::Mode::Classification).pp(pp::pda(0.3F)).make();

  auto j        = spec->to_json();
  auto restored = TrainingSpec::from_json(j);

  EXPECT_EQ(restored->size, 0);
  EXPECT_EQ(restored->seed, 0);
  EXPECT_EQ(restored->threads, 0);
  EXPECT_EQ(restored->max_retries, 3);

  EXPECT_EQ(j, restored->to_json());
}

// ---------------------------------------------------------------------------
// from_json — default values for optional fields
// ---------------------------------------------------------------------------

TEST(TrainingSpec, FromJsonDefaultsOptionalFields) {
  json const j = {
      {"pp", {{"name", "pda"}, {"lambda", 0.3}}},
      {"vars", {{"name", "uniform"}, {"count", 2}}},
      {"cutpoint", {{"name", "mean_of_means"}}},
      {"stop", {{"name", "pure_node"}}},
      {"binarize", {{"name", "largest_gap"}}},
      {"grouping", {{"name", "by_label"}}},
      {"leaf", {{"name", "majority_vote"}}}
  };

  auto spec = TrainingSpec::from_json(j);

  EXPECT_EQ(spec->size, 0);
  EXPECT_EQ(spec->seed, 0);
  EXPECT_EQ(spec->threads, 0);
  EXPECT_EQ(spec->max_retries, 3);
}

// ---------------------------------------------------------------------------
// from_json — missing strategy blocks fall back to Builder::apply_defaults().
// Previously the JSON loader required every strategy key; now the single
// source of truth for default selection lives in `Builder::apply_defaults`,
// and `from_json` only sets the strategies the JSON provides.
// ---------------------------------------------------------------------------

TEST(TrainingSpec, FromJsonMissingPPDefaultsToPda) {
  json const j = {
      {"vars", {{"name", "uniform"}, {"count", 2}}},
      {"cutpoint", {{"name", "mean_of_means"}}},
      {"stop", {{"name", "pure_node"}}},
      {"binarize", {{"name", "largest_gap"}}},
      {"grouping", {{"name", "by_label"}}},
      {"leaf", {{"name", "majority_vote"}}}
  };
  auto spec = TrainingSpec::from_json(j);
  EXPECT_EQ(spec->pp->to_json().at("name"), "pda");
}

TEST(TrainingSpec, FromJsonMissingVarsDefaultsToAll) {
  json const j = {
      {"pp", {{"name", "pda"}, {"lambda", 0.3}}},
      {"cutpoint", {{"name", "mean_of_means"}}},
      {"stop", {{"name", "pure_node"}}},
      {"binarize", {{"name", "largest_gap"}}},
      {"grouping", {{"name", "by_label"}}},
      {"leaf", {{"name", "majority_vote"}}}
  };
  auto spec = TrainingSpec::from_json(j);
  EXPECT_EQ(spec->vars->to_json().at("name"), "all");
}

TEST(TrainingSpec, FromJsonMissingCutpointDefaultsToMeanOfMeans) {
  json const j = {
      {"pp", {{"name", "pda"}, {"lambda", 0.3}}},
      {"vars", {{"name", "uniform"}, {"count", 2}}},
      {"stop", {{"name", "pure_node"}}},
      {"binarize", {{"name", "largest_gap"}}},
      {"grouping", {{"name", "by_label"}}},
      {"leaf", {{"name", "majority_vote"}}}
  };
  auto spec = TrainingSpec::from_json(j);
  EXPECT_EQ(spec->cutpoint->to_json().at("name"), "mean_of_means");
}

TEST(TrainingSpec, FromJsonMissingStopUsesClassificationDefault) {
  json const j = {
      {"mode", "classification"},
      {"pp", {{"name", "pda"}, {"lambda", 0.3}}},
      {"vars", {{"name", "uniform"}, {"count", 2}}},
      {"cutpoint", {{"name", "mean_of_means"}}},
      {"binarize", {{"name", "largest_gap"}}},
      {"grouping", {{"name", "by_label"}}},
      {"leaf", {{"name", "majority_vote"}}}
  };
  auto spec = TrainingSpec::from_json(j);
  EXPECT_EQ(spec->stop->to_json().at("name"), "pure_node");
}

TEST(TrainingSpec, FromJsonMissingStopUsesRegressionDefault) {
  json const j = {
      {"mode", "regression"},
      {"pp", {{"name", "pda"}, {"lambda", 0.3}}},
      {"vars", {{"name", "uniform"}, {"count", 2}}},
      {"cutpoint", {{"name", "mean_of_means"}}},
      {"binarize", {{"name", "disabled"}}},
      {"grouping", {{"name", "by_cutpoint"}}},
      {"leaf", {{"name", "mean_response"}}}
  };
  auto spec      = TrainingSpec::from_json(j);
  auto const& sj = spec->stop->to_json();
  EXPECT_EQ(sj.at("name"), "any");
  ASSERT_TRUE(sj.contains("rules"));
  ASSERT_EQ(sj.at("rules").size(), 2U);
  EXPECT_EQ(sj.at("rules")[0].at("name"), "min_size");
  EXPECT_EQ(sj.at("rules")[1].at("name"), "min_variance");
}

TEST(TrainingSpec, FromJsonMissingBinarizeUsesModeDefault) {
  json const c = {
      {"mode", "classification"},
      {"pp", {{"name", "pda"}, {"lambda", 0.3}}},
      {"vars", {{"name", "uniform"}, {"count", 2}}},
      {"cutpoint", {{"name", "mean_of_means"}}},
      {"stop", {{"name", "pure_node"}}},
      {"grouping", {{"name", "by_label"}}},
      {"leaf", {{"name", "majority_vote"}}}
  };
  EXPECT_EQ(TrainingSpec::from_json(c)->binarization->to_json().at("name"), "largest_gap");

  json const r = {
      {"mode", "regression"},
      {"pp", {{"name", "pda"}, {"lambda", 0.3}}},
      {"vars", {{"name", "uniform"}, {"count", 2}}},
      {"cutpoint", {{"name", "mean_of_means"}}},
      {"stop", {{"name", "min_size"}, {"min_size", 5}}},
      {"grouping", {{"name", "by_cutpoint"}}},
      {"leaf", {{"name", "mean_response"}}}
  };
  EXPECT_EQ(TrainingSpec::from_json(r)->binarization->to_json().at("name"), "disabled");
}

TEST(TrainingSpec, FromJsonMissingGroupingUsesModeDefault) {
  json const c = {
      {"mode", "classification"},
      {"pp", {{"name", "pda"}, {"lambda", 0.3}}},
      {"vars", {{"name", "uniform"}, {"count", 2}}},
      {"cutpoint", {{"name", "mean_of_means"}}},
      {"stop", {{"name", "pure_node"}}},
      {"binarize", {{"name", "largest_gap"}}},
      {"leaf", {{"name", "majority_vote"}}}
  };
  EXPECT_EQ(TrainingSpec::from_json(c)->grouping->to_json().at("name"), "by_label");

  json const r = {
      {"mode", "regression"},
      {"pp", {{"name", "pda"}, {"lambda", 0.3}}},
      {"vars", {{"name", "uniform"}, {"count", 2}}},
      {"cutpoint", {{"name", "mean_of_means"}}},
      {"stop", {{"name", "min_size"}, {"min_size", 5}}},
      {"binarize", {{"name", "disabled"}}},
      {"leaf", {{"name", "mean_response"}}}
  };
  EXPECT_EQ(TrainingSpec::from_json(r)->grouping->to_json().at("name"), "by_cutpoint");
}

TEST(TrainingSpec, FromJsonMissingLeafUsesModeDefault) {
  json const c = {
      {"mode", "classification"},
      {"pp", {{"name", "pda"}, {"lambda", 0.3}}},
      {"vars", {{"name", "uniform"}, {"count", 2}}},
      {"cutpoint", {{"name", "mean_of_means"}}},
      {"stop", {{"name", "pure_node"}}},
      {"binarize", {{"name", "largest_gap"}}},
      {"grouping", {{"name", "by_label"}}}
  };
  EXPECT_EQ(TrainingSpec::from_json(c)->leaf->to_json().at("name"), "majority_vote");

  json const r = {
      {"mode", "regression"},
      {"pp", {{"name", "pda"}, {"lambda", 0.3}}},
      {"vars", {{"name", "uniform"}, {"count", 2}}},
      {"cutpoint", {{"name", "mean_of_means"}}},
      {"stop", {{"name", "min_size"}, {"min_size", 5}}},
      {"binarize", {{"name", "disabled"}}},
      {"grouping", {{"name", "by_cutpoint"}}}
  };
  EXPECT_EQ(TrainingSpec::from_json(r)->leaf->to_json().at("name"), "mean_response");
}

// ---------------------------------------------------------------------------
// display_name must not leak into JSON serialization
// ---------------------------------------------------------------------------

TEST(TrainingSpec, DisplayNameNotInJson) {
  auto spec = TrainingSpec::builder(types::Mode::Classification)
                  .size(5)
                  .threads(2)
                  .pp(pp::pda(0.3F))
                  .vars(vars::uniform(2))
                  .make();
  auto j = spec->to_json();

  EXPECT_FALSE(j["pp"].contains("display_name"));
  EXPECT_FALSE(j["vars"].contains("display_name"));
  EXPECT_FALSE(j["cutpoint"].contains("display_name"));
  EXPECT_FALSE(j["leaf"].contains("display_name"));
}

// ---------------------------------------------------------------------------
// to_json — strategy fields are complete
// ---------------------------------------------------------------------------

TEST(TrainingSpec, ToJsonContainsAllStrategyFields) {
  auto spec = TrainingSpec::builder(types::Mode::Classification)
                  .size(5)
                  .threads(2)
                  .pp(pp::pda(0.3F))
                  .vars(vars::uniform(2))
                  .make();
  auto j = spec->to_json();


  EXPECT_EQ(j["pp"], pp::pda(0.3F)->to_json());
  EXPECT_EQ(j["vars"], vars::uniform(2)->to_json());
  EXPECT_EQ(j["cutpoint"], cutpoint::mean_of_means()->to_json());
  EXPECT_EQ(j["stop"], stop::pure_node()->to_json());
  EXPECT_EQ(j["binarize"], binarize::largest_gap()->to_json());
  EXPECT_EQ(j["grouping"], grouping::by_label()->to_json());
  EXPECT_EQ(j["leaf"], leaf::majority_vote()->to_json());
}

// ---------------------------------------------------------------------------
// supported_modes() — mode/strategy compatibility validation
// ---------------------------------------------------------------------------

TEST(TrainingSpec, RejectsMajorityVoteInRegression) {
  // MajorityVote is classification-only; the builder defaults everything else
  // to regression-compatible.
  auto build = [] {
    return TrainingSpec::builder(types::Mode::Classification)
        .grouping(grouping::by_cutpoint())
        .stop(stop::min_size(5))
        .leaf(leaf::majority_vote())
        .make();
  };

  EXPECT_THROW(build(), ppforest2::UserError);
}

TEST(TrainingSpec, RejectsMeanResponseInClassification) {
  auto build = [] {
    return TrainingSpec::builder(types::Mode::Classification).leaf(leaf::mean_response()).make();
  };

  EXPECT_THROW(build(), ppforest2::UserError);
}

TEST(TrainingSpec, RejectsPureNodeInRegression) {
  auto build = [] {
    return TrainingSpec::builder(types::Mode::Classification)
        .grouping(grouping::by_cutpoint())
        .leaf(leaf::mean_response())
        .stop(stop::pure_node())
        .make();
  };

  EXPECT_THROW(build(), ppforest2::UserError);
}

TEST(TrainingSpec, RejectsMinVarianceInClassification) {
  auto build = [] {
    return TrainingSpec::builder(types::Mode::Classification).stop(stop::min_variance(0.01F)).make();
  };

  EXPECT_THROW(build(), ppforest2::UserError);
}

TEST(TrainingSpec, RejectsByCutpointInClassification) {
  auto build = [] {
    return TrainingSpec::builder(types::Mode::Classification).grouping(grouping::by_cutpoint()).make();
  };

  EXPECT_THROW(build(), ppforest2::UserError);
}

TEST(TrainingSpec, AcceptsCompatibleClassificationStrategies) {
  auto build = [] {
    return TrainingSpec::builder(types::Mode::Classification)
        .stop(stop::pure_node())
        .grouping(grouping::by_label())
        .leaf(leaf::majority_vote())
        .make();
  };

  EXPECT_NO_THROW(build());
}

TEST(TrainingSpec, AcceptsCompatibleRegressionStrategies) {
  auto build = [] {
    return TrainingSpec::builder(types::Mode::Regression)
        .stop(stop::any({stop::min_size(5), stop::min_variance(0.01F)}))
        .grouping(grouping::by_cutpoint())
        .leaf(leaf::mean_response())
        .make();
  };

  EXPECT_NO_THROW(build());
}

TEST(TrainingSpec, CompositeStopIntersectsChildModes) {
  // stop::any(min_size, pure_node) = classification only (pure_node restricts it).
  auto rule  = stop::any({stop::min_size(5), stop::pure_node()});
  auto modes = rule->supported_modes();

  EXPECT_TRUE(modes.count(types::Mode::Classification) > 0);
  EXPECT_TRUE(modes.count(types::Mode::Regression) == 0);

  // Using it in regression mode should fail.
  auto build = [&] {
    return TrainingSpec::builder(types::Mode::Classification)
        .grouping(grouping::by_cutpoint())
        .leaf(leaf::mean_response())
        .stop(rule)
        .make();
  };

  EXPECT_THROW(build(), ppforest2::UserError);
}

TEST(TrainingSpec, CompositeStopOnlyRegressionChildrenSupportsRegression) {
  auto rule  = stop::any({stop::min_size(5), stop::min_variance(0.01F)});
  auto modes = rule->supported_modes();

  // min_size supports both; min_variance supports only regression;
  // intersection is {Regression}.
  EXPECT_TRUE(modes.count(types::Mode::Regression) > 0);
  EXPECT_TRUE(modes.count(types::Mode::Classification) == 0);
}

TEST(TrainingSpec, MinSizeSupportsBothModes) {
  auto modes = stop::min_size(5)->supported_modes();
  EXPECT_TRUE(modes.count(types::Mode::Classification) > 0);
  EXPECT_TRUE(modes.count(types::Mode::Regression) > 0);
}

TEST(TrainingSpec, ByLabelSupportsClassificationOnly) {
  auto modes = grouping::by_label()->supported_modes();
  EXPECT_TRUE(modes.count(types::Mode::Classification) > 0);
  EXPECT_TRUE(modes.count(types::Mode::Regression) == 0);
}

TEST(TrainingSpec, AcceptsDisabledBinarizeInRegression) {
  // `binarize::Disabled` is a mode-agnostic placeholder for specs where
  // binarize never fires (regression's `ByCutpoint` grouping always
  // yields 2 groups). A regression spec built with `Disabled` passes
  // mode validation. The builder's mode-aware default also resolves
  // to `Disabled` here — this test pins both the explicit and implicit
  // paths.
  auto build_explicit = [] {
    return TrainingSpec::builder(types::Mode::Regression)
        .grouping(grouping::by_cutpoint())
        .leaf(leaf::mean_response())
        .stop(stop::min_size(5))
        .binarization(binarize::disabled())
        .make();
  };

  auto build_default = [] {
    return TrainingSpec::builder(types::Mode::Regression)
        .grouping(grouping::by_cutpoint())
        .leaf(leaf::mean_response())
        .stop(stop::min_size(5))
        .make(); // binarization resolves to Disabled via the builder default
  };

  EXPECT_NO_THROW(build_explicit());
  EXPECT_NO_THROW(build_default());
}

TEST(TrainingSpec, RejectsLargestGapInRegression) {
  // `LargestGap` is classification-only; regression specs must use
  // `Disabled` (or another regression-compatible binarizer). This
  // replaces the old "binarize is exempt" behavior — the placeholder
  // lets binarize participate in mode validation like every other
  // strategy family.
  auto build = [] {
    return TrainingSpec::builder(types::Mode::Classification)
        .grouping(grouping::by_cutpoint())
        .leaf(leaf::mean_response())
        .stop(stop::min_size(5))
        .binarization(binarize::largest_gap())
        .make();
  };

  EXPECT_THROW(build(), ppforest2::UserError);
}

// ---------------------------------------------------------------------------
// Builder::apply_defaults — the single source of truth for mode-aware
// default strategies. Both CLI and R bindings pass only the strategies
// the user explicitly supplied and rely on this method to fill in the
// rest, so the test pins all four defaults per mode and verifies the
// idempotence + user-override semantics.
// ---------------------------------------------------------------------------

TEST(TrainingSpec, ApplyDefaultsClassification) {
  auto b = TrainingSpec::builder(types::Mode::Classification);
  b.apply_defaults();

  EXPECT_EQ(b.config.pp->to_json().at("name"), "pda");
  EXPECT_EQ(b.config.vars->to_json().at("name"), "all");
  EXPECT_EQ(b.config.cutpoint->to_json().at("name"), "mean_of_means");
  EXPECT_EQ(b.config.stop->to_json().at("name"), "pure_node");
  EXPECT_EQ(b.config.binarization->to_json().at("name"), "largest_gap");
  EXPECT_EQ(b.config.grouping->to_json().at("name"), "by_label");
  EXPECT_EQ(b.config.leaf->to_json().at("name"), "majority_vote");
}

TEST(TrainingSpec, ApplyDefaultsRegression) {
  auto b = TrainingSpec::builder(types::Mode::Regression);
  b.apply_defaults();

  EXPECT_EQ(b.config.pp->to_json().at("name"), "pda");
  EXPECT_EQ(b.config.vars->to_json().at("name"), "all");
  EXPECT_EQ(b.config.cutpoint->to_json().at("name"), "mean_of_means");
  // Regression stop: composite `any` of min_size(5) and min_variance(0.01).
  auto const& sj = b.config.stop->to_json();
  EXPECT_EQ(sj.at("name"), "any");
  ASSERT_EQ(sj.at("rules").size(), 2U);
  EXPECT_EQ(sj.at("rules")[0].at("name"), "min_size");
  EXPECT_EQ(sj.at("rules")[0].at("min_size").get<int>(), 5);
  EXPECT_EQ(sj.at("rules")[1].at("name"), "min_variance");
  EXPECT_FLOAT_EQ(sj.at("rules")[1].at("threshold").get<float>(), 0.01F);
  // Regression's by_cutpoint grouping always emits 2 groups, so the
  // binarize default is `disabled`.
  EXPECT_EQ(b.config.binarization->to_json().at("name"), "disabled");
  EXPECT_EQ(b.config.grouping->to_json().at("name"), "by_cutpoint");
  EXPECT_EQ(b.config.leaf->to_json().at("name"), "mean_response");
}

TEST(TrainingSpec, ApplyDefaultsIsIdempotent) {
  // Re-calling apply_defaults must not clobber user-set strategies.
  auto b = TrainingSpec::builder(types::Mode::Classification).stop(stop::max_depth(7));
  b.apply_defaults().apply_defaults();
  auto const& sj = b.config.stop->to_json();
  EXPECT_EQ(sj.at("name"), "max_depth");
  EXPECT_EQ(sj.at("max_depth").get<int>(), 7);
}

TEST(TrainingSpec, ApplyDefaultsLeavesUserChoicesIntact) {
  // Each user-supplied strategy must survive default resolution untouched.
  auto b = TrainingSpec::builder(types::Mode::Regression)
               .leaf(leaf::mean_response())
               .grouping(grouping::by_cutpoint())
               .stop(stop::min_size(20))
               .binarization(binarize::disabled());
  b.apply_defaults();

  EXPECT_EQ(b.config.leaf->to_json().at("name"), "mean_response");
  EXPECT_EQ(b.config.grouping->to_json().at("name"), "by_cutpoint");
  EXPECT_EQ(b.config.stop->to_json().at("name"), "min_size");
  EXPECT_EQ(b.config.stop->to_json().at("min_size").get<int>(), 20);
  EXPECT_EQ(b.config.binarization->to_json().at("name"), "disabled");
}
