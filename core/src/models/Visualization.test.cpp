#include <gtest/gtest.h>
#include <Eigen/Dense>

#include "models/Visualization.hpp"
#include "models/ClassificationTree.hpp"
#include "models/TreeBranch.hpp"
#include "models/TreeLeaf.hpp"
#include "models/TrainingSpec.hpp"
#include "test/Projectors.hpp"
#include "stats/Stats.hpp"

using namespace ppforest2;
using namespace ppforest2::stats;
using namespace ppforest2::test;
using namespace ppforest2::types;
using namespace ppforest2::viz;
using namespace ppforest2::pp;

namespace {
  ClassificationTree test_tree() {
    return ClassificationTree(
        TreeBranch::make(
            as_projector({0.7F, 0.3F, 0.5F, 0.1F}),
            1.5F,
            TreeLeaf::make(0),
            TreeBranch::make(
                as_projector({0.2F, 0.8F, 0.1F, 0.4F}), 2.0F, TreeLeaf::make(1), TreeLeaf::make(2), {1, 2}
            ),
            {0, 1, 2}
        ),
        TrainingSpec::builder(types::Mode::Classification).pp(pp::pda(0.5F)).make(),
        ClassificationTree::Groups{0, 1, 2}
    );
  }

  // Make a deeper tree (depth 4) to stress-test memory management
  ClassificationTree deep_tree() {
    return ClassificationTree(
        TreeBranch::make(
            as_projector({0.4F, 0.6F}),
            1.8F,
            TreeBranch::make(
                as_projector({0.5F, 0.5F}),
                1.5F,
                TreeBranch::make(as_projector({0.6F, 0.4F}), 1.0F, TreeLeaf::make(0), TreeLeaf::make(1), {0, 1}),
                TreeLeaf::make(1),
                {0, 1}
            ),
            TreeBranch::make(
                as_projector({0.8F, 0.2F}),
                2.5F,
                TreeLeaf::make(2),
                TreeBranch::make(as_projector({0.3F, 0.7F}), 2.0F, TreeLeaf::make(2), TreeLeaf::make(3), {2, 3}),
                {2, 3}
            ),
            {0, 1, 2, 3}
        ),
        TrainingSpec::builder(types::Mode::Classification).pp(pp::pda(0.5F)).make(),
        ClassificationTree::Groups{0, 1, 2, 3}
    );
  }
}


class VisualizationTest : public ::testing::Test {
protected:
  FeatureMatrix x;
  OutcomeVector y;
  ClassificationTree tree = test_tree();

  void SetUp() override {
    // 30 observations, 4 features
    x.resize(30, 4);
    y.resize(30);

    RNG rng(0);

    for (int i = 0; i < 30; ++i) {
      for (int j = 0; j < 4; ++j) {
        x(i, j) = static_cast<Feature>(i * 0.1F + j * 0.5F + (rng() % 100) * 0.01F);
      }

      y(i) = i % 3;
    }
  }
};

TEST_F(VisualizationTest, NodeDataVisitorCollectsAllNodes) {
  NodeDataVisitor visitor(x, y);
  tree.root->accept(visitor);

  // Tree has 5 nodes: root, leaf0, right_cond, leaf1, leaf2
  ASSERT_EQ(visitor.nodes.size(), 5u);

  // Root is internal
  EXPECT_FALSE(visitor.nodes[0].is_leaf);
  EXPECT_EQ(visitor.nodes[0].depth, 0);
  EXPECT_EQ(visitor.nodes[0].projected_values.size(), 30u);
  EXPECT_EQ(visitor.nodes[0].groups.size(), 30u);

  // Check that at least one leaf exists
  bool found_leaf = false;

  for (auto const& nd : visitor.nodes) {
    if (nd.is_leaf) {
      found_leaf = true;
      EXPECT_TRUE(nd.projected_values.empty());
    }
  }

  EXPECT_TRUE(found_leaf);
}

TEST_F(VisualizationTest, BoundaryVisitorCollectsSegments) {
  BoundaryVisitor visitor(0, 1, {}, -1.0F, 5.0F, -1.0F, 5.0F);

  tree.root->accept(visitor);

  // Should have at least 1 boundary segment (one per internal node that clips)
  EXPECT_GT(visitor.segments.size(), 0U);

  for (auto const& seg : visitor.segments) {
    EXPECT_GE(seg.depth, 0);
  }
}

TEST_F(VisualizationTest, BoundaryVisitorWithFixedVars) {
  std::vector<std::pair<int, Feature>> fixed = {{2, 1.5F}, {3, 0.5F}};
  BoundaryVisitor visitor(0, 1, fixed, -1.0F, 5.0F, -1.0F, 5.0F);

  tree.root->accept(visitor);

  // Should still produce segments
  EXPECT_GT(visitor.segments.size(), 0U);
}

TEST_F(VisualizationTest, RegionVisitorCollectsPolygons) {
  RegionVisitor visitor(0, 1, {}, -1.0F, 5.0F, -1.0F, 5.0F);

  tree.root->accept(visitor);

  // Should have one region per leaf (3 leaves)
  EXPECT_EQ(visitor.regions.size(), 3U);

  for (auto const& region : visitor.regions) {
    // Each region should be a polygon with at least 3 vertices
    EXPECT_GE(region.vertices.size(), 3U);
    // Predicted group should be valid
    EXPECT_GE(region.predicted_group, 0);
    EXPECT_LE(region.predicted_group, 2);
  }
}

TEST_F(VisualizationTest, RegionVisitorWithFixedVars) {
  std::vector<std::pair<int, Feature>> fixed = {{2, 1.5F}, {3, 0.5F}};
  RegionVisitor visitor(0, 1, fixed, -1.0F, 5.0F, -1.0F, 5.0F);

  tree.root->accept(visitor);

  EXPECT_EQ(visitor.regions.size(), 3U);
}

TEST_F(VisualizationTest, ClipPolygonHalfspaceBasic) {
  // Unit square
  Polygon const square = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};

  // Clip with x < 0.5
  FeatureVector normal(2);
  normal << 1.0F, 0.0F;
  Polygon const clipped = clip_polygon_halfspace(square, normal, 0.5F, true);

  EXPECT_EQ(clipped.size(), 4U);

  for (auto const& v : clipped) {
    EXPECT_LT(v.first, 0.5F + 1e-6F);
  }
}

TEST_F(VisualizationTest, ClipPolygonHalfspaceEmpty) {
  Polygon const square = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};

  // Clip with x < -1 (nothing inside)
  FeatureVector normal(2);
  normal << 1.0F, 0.0F;
  Polygon const clipped = clip_polygon_halfspace(square, normal, -1.0F, true);

  EXPECT_TRUE(clipped.empty());
}

TEST_F(VisualizationTest, ComputeTreeLayoutProducesCorrectCounts) {
  TreeLayout layout = compute_tree_layout(*tree.root);

  // 5 nodes, 4 edges (each internal node has 2 children edges)
  EXPECT_EQ(layout.nodes.size(), 5U);
  EXPECT_EQ(layout.edges.size(), 4U);

  // Check that nodes have unique positions
  for (size_t i = 0; i < layout.nodes.size(); ++i) {
    for (size_t j = i + 1; j < layout.nodes.size(); ++j) {
      bool const same_pos = (layout.nodes[i].x == layout.nodes[j].x && layout.nodes[i].y == layout.nodes[j].y);
      EXPECT_FALSE(same_pos) << "Nodes " << i << " and " << j << " overlap";
    }
  }
}

TEST_F(VisualizationTest, ComputeTreeLayoutLeafAndInternalFlags) {
  TreeLayout const layout = compute_tree_layout(*tree.root);

  int n_internal = 0;
  int n_leaf     = 0;

  for (auto const& node : layout.nodes) {
    if (node.is_leaf) {
      n_leaf++;
    } else {
      n_internal++;
    }
  }

  EXPECT_EQ(n_internal, 2);
  EXPECT_EQ(n_leaf, 3);
}

TEST_F(VisualizationTest, ComputeTreeLayoutEdgesHaveLabels) {
  TreeLayout const layout = compute_tree_layout(*tree.root);

  for (auto const& edge : layout.edges) {
    EXPECT_FALSE(edge.label.empty());
  }
}

// Deep tree tests stress memory management under ASan
TEST(VisualizationDeepTree, NodeDataVisitor) {
  ClassificationTree deep = deep_tree();

  FeatureMatrix x(20, 2);
  OutcomeVector y(20);

  for (int i = 0; i < 20; ++i) {
    x(i, 0) = static_cast<Feature>(i) * 0.2F;
    x(i, 1) = static_cast<Feature>(i) * 0.3F;
    y(i)    = i % 4;
  }

  NodeDataVisitor visitor(x, y);
  deep.root->accept(visitor);

  // 11 nodes: 5 internal + 6 leaves
  EXPECT_EQ(visitor.nodes.size(), 11U);
}

TEST(VisualizationDeepTree, BoundaryVisitor) {
  ClassificationTree deep = deep_tree();

  BoundaryVisitor visitor(0, 1, {}, -1.0F, 10.0F, -1.0F, 10.0F);
  deep.root->accept(visitor);

  EXPECT_GT(visitor.segments.size(), 0U);
}

TEST(VisualizationDeepTree, RegionVisitor) {
  ClassificationTree deep = deep_tree();

  RegionVisitor visitor(0, 1, {}, -1.0F, 10.0F, -1.0F, 10.0F);
  deep.root->accept(visitor);

  // Should have one region per leaf (6 leaves)
  EXPECT_EQ(visitor.regions.size(), 6U);
}

TEST(VisualizationDeepTree, ComputeTreeLayout) {
  ClassificationTree const deep = deep_tree();
  TreeLayout const layout       = compute_tree_layout(*deep.root);

  // 11 nodes, 10 edges
  EXPECT_EQ(layout.nodes.size(), 11U);
  EXPECT_EQ(layout.edges.size(), 10U);
}

// Test repeated visitor creation/destruction (stress allocator)
TEST(VisualizationStress, RepeatedVisitorCalls) {
  ClassificationTree tree = test_tree();

  for (int iter = 0; iter < 50; ++iter) {
    {
      BoundaryVisitor bv(0, 1, {}, -1.0F, 5.0F, -1.0F, 5.0F);
      tree.root->accept(bv);
      ASSERT_GT(bv.segments.size(), 0U);
    }

    {
      RegionVisitor rv(0, 1, {}, -1.0F, 5.0F, -1.0F, 5.0F);
      tree.root->accept(rv);
      ASSERT_EQ(rv.regions.size(), 3U);
    }

    {
      TreeLayout const layout = compute_tree_layout(*tree.root);
      ASSERT_EQ(layout.nodes.size(), 5U);
    }
  }
}

// Test with many different variable pairs (like pairwise plot does)
TEST_F(VisualizationTest, PairwiseBoundaryAndRegions) {
  int const p = 4;

  for (int i = 0; i < p; ++i) {
    for (int j = i + 1; j < p; ++j) {
      std::vector<std::pair<int, Feature>> fixed;

      for (int k = 0; k < p; ++k) {
        if (k != i && k != j) {
          fixed.push_back({k, 1.0F});
        }
      }

      BoundaryVisitor bv(i, j, fixed, -1.0F, 5.0F, -1.0F, 5.0F);
      tree.root->accept(bv);
      EXPECT_GT(bv.segments.size(), 0U) << "No segments for pair (" << i << ", " << j << ")";

      RegionVisitor rv(i, j, fixed, -1.0F, 5.0F, -1.0F, 5.0F);
      tree.root->accept(rv);
      EXPECT_EQ(rv.regions.size(), 3U) << "Wrong region count for pair (" << i << ", " << j << ")";
    }
  }
}
