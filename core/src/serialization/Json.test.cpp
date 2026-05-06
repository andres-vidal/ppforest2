/**
 * @file Json.test.cpp
 * @brief Tests for JSON serialization round-trips.
 *
 * Uses committed golden files as fixtures — loads model JSON, deserializes,
 * re-serializes, and compares.  No training is performed.
 */
#include <gtest/gtest.h>

#include "serialization/Json.hpp"
#include "models/Bagged.hpp"
#include "utils/Invariant.hpp"

#include <fstream>

using namespace ppforest2;
using namespace ppforest2::types;
using namespace ppforest2::stats;
using namespace ppforest2::serialization;

using json = nlohmann::json;

#ifndef PPFOREST2_GOLDEN_DIR
#error "PPFOREST2_GOLDEN_DIR must be defined"
#endif

#ifndef PPFOREST2_FIXTURES_DIR
#error "PPFOREST2_FIXTURES_DIR must be defined"
#endif

static const std::string GOLDEN_DIR   = PPFOREST2_GOLDEN_DIR;
static std::string const FIXTURES_DIR = PPFOREST2_FIXTURES_DIR;

namespace {
  json load_golden(std::string const& path) {
    std::ifstream in(path);
    invariant(in.is_open(), "Required golden file missing: " + path);
    return json::parse(in);
  }
}

TEST(JsonRoundTrip, Tree) {
  auto golden = load_golden(GOLDEN_DIR + "/iris/tree-pda-s0.json");
  auto e      = golden.get<Export<Tree::Ptr>>();

  json j = to_json(*e.model, e.groups);

  ASSERT_EQ(golden["model"], j) << "Tree JSON should be identical after round-trip";
}

TEST(JsonRoundTrip, Forest) {
  auto golden = load_golden(GOLDEN_DIR + "/iris/forest-pda-n5-s0.json");
  auto e      = golden.get<Export<Forest::Ptr>>();

  json j = to_json(*e.model, e.groups);

  ASSERT_EQ(golden["model"], j) << "Forest JSON should be identical after round-trip";
}

TEST(JsonRoundTrip, ModelDispatchTree) {
  auto golden = load_golden(GOLDEN_DIR + "/iris/tree-pda-s0.json");

  auto e = golden.get<Export<Model::Ptr>>();
  ASSERT_NE(e.model, nullptr);

  json j = to_json(*e.model);
  ASSERT_EQ(j["model_type"], "tree");
}

TEST(JsonRoundTrip, ModelDispatchForest) {
  auto golden = load_golden(GOLDEN_DIR + "/iris/forest-pda-n5-s0.json");

  auto e = golden.get<Export<Model::Ptr>>();
  ASSERT_NE(e.model, nullptr);

  json j = to_json(*e.model);
  ASSERT_EQ(j["model_type"], "forest");
}

TEST(JsonRoundTrip, ConfusionMatrix) {
  GroupIdVector predictions(6);
  predictions << 0, 0, 1, 1, 2, 2;

  GroupIdVector actual(6);
  actual << 0, 1, 1, 1, 2, 0;

  ConfusionMatrix cm(predictions, actual);
  json j = to_json(cm);

  ASSERT_TRUE(j.contains("matrix"));
  ASSERT_TRUE(j.contains("labels"));
  ASSERT_TRUE(j.contains("group_errors"));

  auto matrix = j["matrix"].get<std::vector<std::vector<int>>>();
  ASSERT_EQ(matrix.size(), 3);
  ASSERT_EQ(matrix[0].size(), 3);
}

// ---------------------------------------------------------------------------
// Labeled serialization tests
// ---------------------------------------------------------------------------

TEST(JsonLabeled, TreeLeafValuesAreStrings) {
  auto golden = load_golden(GOLDEN_DIR + "/iris/tree-pda-s0.json");
  auto e      = golden.get<Export<Tree::Ptr>>();
  json j      = to_json(*e.model, e.groups);

  // Walk to a leaf
  json const* node = &j["root"];

  while (node->contains("lower"))
    node = &(*node)["lower"];

  ASSERT_TRUE((*node)["value"].is_string());
  auto value = (*node)["value"].get<std::string>();
  EXPECT_TRUE(std::find(e.groups.begin(), e.groups.end(), value) != e.groups.end())
      << "Leaf value '" << value << "' not in group_names";
}

TEST(JsonLabeled, TreeGroupsAreStrings) {
  auto golden = load_golden(GOLDEN_DIR + "/iris/tree-pda-s0.json");
  auto e      = golden.get<Export<Tree::Ptr>>();
  json j      = to_json(*e.model, e.groups);

  auto root_groups = j["root"]["groups"].get<std::vector<std::string>>();
  EXPECT_EQ(root_groups.size(), e.groups.size());

  for (auto const& g : root_groups) {
    EXPECT_TRUE(std::find(e.groups.begin(), e.groups.end(), g) != e.groups.end())
        << "Group '" << g << "' not in group_names";
  }
}

TEST(JsonLabeled, IntegerFormatOmitsGroupNames) {
  auto golden = load_golden(GOLDEN_DIR + "/iris/tree-pda-s0.json");
  auto e      = golden.get<Export<Tree::Ptr>>();
  json j      = to_json(*e.model);

  // Walk to a leaf
  json const* node = &j["root"];

  while (node->contains("lower"))
    node = &(*node)["lower"];

  ASSERT_TRUE((*node)["value"].is_number());
}

TEST(JsonLabeled, ConfusionMatrixLabelsAreStrings) {
  GroupIdVector predictions(6);
  predictions << 0, 0, 1, 1, 2, 2;

  GroupIdVector actual(6);
  actual << 0, 1, 1, 1, 2, 0;

  Names names = {"setosa", "versicolor", "virginica"};
  ConfusionMatrix cm(predictions, actual);
  json j = to_json(cm, names);

  auto labels = j["labels"].get<std::vector<std::string>>();
  EXPECT_EQ(labels, names);
}

TEST(JsonLabeled, ForestRoundTrip) {
  auto golden = load_golden(GOLDEN_DIR + "/iris/forest-pda-n5-s0.json");
  auto e      = golden.get<Export<Forest::Ptr>>();

  json j = to_json(*e.model, e.groups);

  ASSERT_EQ(golden["model"], j) << "Forest labeled round-trip should be identical";
}

TEST(JsonLabeled, LabeledAndIntegerLeavesAgreeOnPredictedGroup) {
  // The labeled and integer-format writers produce the same logical model;
  // the only difference is how leaf values are rendered (label string vs.
  // integer index). Walk both trees in lockstep and verify that each
  // labeled leaf's string value matches the integer leaf indexed through
  // `group_names`.
  auto golden = load_golden(GOLDEN_DIR + "/iris/tree-pda-s0.json");
  auto e      = golden.get<Export<Tree::Ptr>>();

  json labeled   = to_json(*e.model, e.groups);
  json unlabeled = to_json(*e.model);

  std::function<void(json const&, json const&)> walk = [&](json const& l, json const& u) {
    if (l.contains("value")) {
      ASSERT_TRUE(u.contains("value")) << "tree shape diverged";
      ASSERT_TRUE(l["value"].is_string()) << "labeled leaf must carry a string label";
      ASSERT_TRUE(u["value"].is_number()) << "integer-format leaf must carry an integer index";
      EXPECT_EQ((e.groups)[u["value"].get<int>()], l["value"].get<std::string>())
          << "labeled / integer leaf value mismatch";
    } else {
      ASSERT_TRUE(l.contains("lower")) << "labeled branch missing 'lower'";
      walk(l["lower"], u["lower"]);
      walk(l["upper"], u["upper"]);
    }
  };
  walk(labeled["root"], unlabeled["root"]);
}

TEST(JsonRoundTrip, ModelFromJsonFile) {
  std::string model_path = FIXTURES_DIR + "/model.json";
  std::ifstream in(model_path);

  invariant(in.is_open(), "model.json not found at " + model_path);

  json j = json::parse(in);
  in.close();

  ASSERT_NO_THROW({
    auto e = j.get<Export<Model::Ptr>>();
    ASSERT_NE(e.model, nullptr);
  });
}

// ---------------------------------------------------------------------------
// Serialization structure tests — verify JSON shape without training
// ---------------------------------------------------------------------------

TEST(JsonStructure, TreeAlwaysHasDegenerate) {
  auto golden = load_golden(GOLDEN_DIR + "/iris/tree-pda-s0.json");
  auto e      = golden.get<Export<Tree::Ptr>>();
  json j      = to_json(*e.model, e.groups);

  EXPECT_TRUE(j.contains("degenerate"));
  EXPECT_FALSE(j["degenerate"].get<bool>());
}

TEST(JsonStructure, ForestAlwaysHasDegenerate) {
  auto golden = load_golden(GOLDEN_DIR + "/iris/forest-pda-n5-s0.json");
  auto e      = golden.get<Export<Forest::Ptr>>();
  json j      = to_json(*e.model, e.groups);

  EXPECT_TRUE(j.contains("degenerate"));
  EXPECT_FALSE(j["degenerate"].get<bool>());
}

TEST(JsonStructure, ConfusionMatrixRoundTrip) {
  GroupIdVector predictions(6);
  predictions << 0, 0, 1, 1, 2, 2;

  GroupIdVector actual(6);
  actual << 0, 1, 1, 1, 2, 0;

  ConfusionMatrix cm(predictions, actual);
  json j = to_json(cm);

  auto restored = j.get<ConfusionMatrix>();

  EXPECT_EQ(restored.values.rows(), cm.values.rows());
  EXPECT_EQ(restored.values.cols(), cm.values.cols());
  EXPECT_EQ(restored.values, cm.values);
}

TEST(JsonStructure, ForestSampleIndicesRoundTrip) {
  auto golden = load_golden(GOLDEN_DIR + "/iris/forest-pda-n5-s0.json");
  auto e      = golden.get<Export<Forest::Ptr>>();

  json j = to_json(*e.model, e.groups);

  for (size_t i = 0; i < e.model->trees.size(); ++i) {
    auto const* bt = dynamic_cast<BaggedTree const*>(e.model->trees[i].get());
    ASSERT_NE(bt, nullptr) << "Tree " << i << " should be a BaggedTree";
    EXPECT_FALSE(bt->sample_indices.empty()) << "Tree " << i << " should have sample_indices";

    auto rt_indices = j["trees"][i]["sample_indices"].get<std::vector<int>>();
    EXPECT_EQ(rt_indices, bt->sample_indices) << "Tree " << i << " sample_indices should round-trip";
  }
}

TEST(JsonStructure, VariableImportanceRoundTrip) {
  VariableImportance vi;
  vi.scale       = types::FeatureVector::Ones(4);
  vi.projections = types::FeatureVector::Random(4);

  json j = to_json(vi);

  auto restored = j.get<VariableImportance>();

  EXPECT_EQ(restored.projections.size(), vi.projections.size());
  EXPECT_EQ(restored.scale.size(), vi.scale.size());
}

// ---------------------------------------------------------------------------
// Export round-trip tests — full JSON (model + config + meta + metrics)
// ---------------------------------------------------------------------------

TEST(ExportRoundTrip, TreePreservesModelAndMeta) {
  auto golden = load_golden(GOLDEN_DIR + "/iris/tree-pda-s0.json");
  auto e      = golden.get<Export<Model::Ptr>>();

  json j = e.to_json();

  EXPECT_TRUE(j.contains("model_type"));
  EXPECT_EQ(j["model_type"], "tree");
  EXPECT_TRUE(j.contains("config"));
  EXPECT_TRUE(j.contains("meta"));
  EXPECT_EQ(j["meta"]["groups"], golden["meta"]["groups"]);
  EXPECT_EQ(j["meta"]["observations"], golden["meta"]["observations"]);
  EXPECT_EQ(j["meta"]["features"], golden["meta"]["features"]);
  EXPECT_EQ(j["meta"]["feature_names"], golden["meta"]["feature_names"]);
  EXPECT_EQ(j["model"], golden["model"]);
}

TEST(ExportRoundTrip, ForestPreservesModelAndMeta) {
  auto golden = load_golden(GOLDEN_DIR + "/iris/forest-pda-n5-s0.json");
  auto e      = golden.get<Export<Model::Ptr>>();

  json j = e.to_json();

  EXPECT_TRUE(j.contains("model_type"));
  EXPECT_EQ(j["model_type"], "forest");
  EXPECT_TRUE(j.contains("config"));
  EXPECT_TRUE(j.contains("meta"));
  EXPECT_EQ(j["meta"]["groups"], golden["meta"]["groups"]);
  EXPECT_EQ(j["model"], golden["model"]);
}

TEST(ExportRoundTrip, TreePreservesConfig) {
  auto golden = load_golden(GOLDEN_DIR + "/iris/tree-pda-s0.json");
  auto e      = golden.get<Export<Model::Ptr>>();

  json j = e.to_json();

  EXPECT_EQ(j["config"]["pp"]["method"], golden["config"]["pp"]["method"]);
  EXPECT_EQ(j["config"]["pp"]["lambda"], golden["config"]["pp"]["lambda"]);
}

TEST(ExportRoundTrip, TreePreservesMetrics) {
  auto golden = load_golden(GOLDEN_DIR + "/iris/tree-pda-s0.json");
  auto e      = golden.get<Export<Model::Ptr>>();

  json j = e.to_json();

  EXPECT_TRUE(j.contains("training_metrics"));
  EXPECT_EQ(j["training_metrics"], golden["training_metrics"]);
  EXPECT_TRUE(j.contains("variable_importance"));
  EXPECT_EQ(j["variable_importance"], golden["variable_importance"]);
}

TEST(ExportRoundTrip, ForestPreservesMetrics) {
  auto golden = load_golden(GOLDEN_DIR + "/iris/forest-pda-n5-s0.json");
  auto e      = golden.get<Export<Model::Ptr>>();

  json j = e.to_json();

  EXPECT_TRUE(j.contains("training_metrics"));
  EXPECT_EQ(j["training_metrics"], golden["training_metrics"]);
  EXPECT_TRUE(j.contains("variable_importance"));
  EXPECT_EQ(j["variable_importance"], golden["variable_importance"]);
  EXPECT_TRUE(j.contains("oob_metrics"));
  EXPECT_EQ(j["oob_metrics"], golden["oob_metrics"]);
}

TEST(ExportRoundTrip, FullRoundTripTreeIdentical) {
  auto golden = load_golden(GOLDEN_DIR + "/iris/tree-pda-s0.json");
  auto e      = golden.get<Export<Model::Ptr>>();

  json j = e.to_json();

  // Re-deserialize and re-serialize — should be stable
  auto e2    = j.get<Export<Model::Ptr>>();
  json again = e2.to_json();

  EXPECT_EQ(j["model"], again["model"]);
  EXPECT_EQ(j["config"], again["config"]);
  EXPECT_EQ(j["meta"], again["meta"]);
}

TEST(ExportRoundTrip, FullRoundTripForestIdentical) {
  auto golden = load_golden(GOLDEN_DIR + "/iris/forest-pda-n5-s0.json");
  auto e      = golden.get<Export<Model::Ptr>>();

  json j = e.to_json();

  auto e2    = j.get<Export<Model::Ptr>>();
  json again = e2.to_json();

  EXPECT_EQ(j["model"], again["model"]);
  EXPECT_EQ(j["config"], again["config"]);
  EXPECT_EQ(j["meta"], again["meta"]);
}

// ---------------------------------------------------------------------------
// Eigen ↔ JSON ADL serializer
// ---------------------------------------------------------------------------

TEST(EigenJson, ColumnVectorToJsonIsFlatArray) {
  FeatureVector v(3);
  v << 1.0F, 2.0F, 3.5F;

  json j = v;

  ASSERT_TRUE(j.is_array());
  EXPECT_EQ(j.size(), 3u);
  EXPECT_FLOAT_EQ(j[0].get<float>(), 1.0F);
  EXPECT_FLOAT_EQ(j[2].get<float>(), 3.5F);
}

TEST(EigenJson, ColumnVectorRoundTrip) {
  FeatureVector v(4);
  v << -1.0F, 0.0F, 0.5F, 100.0F;

  json j        = v;
  auto restored = j.get<FeatureVector>();

  ASSERT_EQ(restored.size(), v.size());
  for (Eigen::Index i = 0; i < v.size(); ++i) {
    EXPECT_FLOAT_EQ(restored(i), v(i));
  }
}

TEST(EigenJson, MatrixToJsonIsNestedArray) {
  FeatureMatrix m(2, 3);
  m << 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F;

  json j = m;

  ASSERT_TRUE(j.is_array());
  EXPECT_EQ(j.size(), 2u);
  ASSERT_TRUE(j[0].is_array());
  EXPECT_EQ(j[0].size(), 3u);
  EXPECT_FLOAT_EQ(j[0][1].get<float>(), 2.0F);
  EXPECT_FLOAT_EQ(j[1][2].get<float>(), 6.0F);
}

TEST(EigenJson, MatrixRoundTrip) {
  FeatureMatrix m(3, 2);
  m << 1.5F, -1.5F, 0.0F, 4.25F, 7.0F, -8.0F;

  json j        = m;
  auto restored = j.get<FeatureMatrix>();

  ASSERT_EQ(restored.rows(), m.rows());
  ASSERT_EQ(restored.cols(), m.cols());
  for (Eigen::Index i = 0; i < m.rows(); ++i) {
    for (Eigen::Index k = 0; k < m.cols(); ++k) {
      EXPECT_FLOAT_EQ(restored(i, k), m(i, k));
    }
  }
}

TEST(EigenJson, IntegerVectorRoundTrip) {
  GroupIdVector v(3);
  v << 0, 2, 1;

  json j        = v;
  auto restored = j.get<GroupIdVector>();

  ASSERT_EQ(restored.size(), v.size());
  for (Eigen::Index i = 0; i < v.size(); ++i) {
    EXPECT_EQ(restored(i), v(i));
  }
}

TEST(EigenJson, EmptyVectorRoundTrip) {
  FeatureVector v(0);

  json j        = v;
  auto restored = j.get<FeatureVector>();

  EXPECT_EQ(j.size(), 0u);
  EXPECT_EQ(restored.size(), 0);
}

TEST(EigenJson, EmptyMatrixRoundTrip) {
  FeatureMatrix m(0, 0);

  json j        = m;
  auto restored = j.get<FeatureMatrix>();

  EXPECT_EQ(restored.rows(), 0);
  EXPECT_EQ(restored.cols(), 0);
}
