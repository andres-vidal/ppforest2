#include <gtest/gtest.h>

#include "models/TreeLeaf.hpp"

using namespace ppforest2;

TEST(TreeLeaf, EqualsEqualResponses) {
  TreeLeaf const r1(1);
  TreeLeaf const r2(1);

  ASSERT_TRUE(r1 == r2);
}

TEST(TreeLeaf, EqualsDifferentResponses) {
  TreeLeaf const r1(1);
  TreeLeaf const r2(2);

  ASSERT_FALSE(r1 == r2);
}

TEST(IsLeaf, TrueForTreeLeaf) {
  TreeLeaf const leaf(0);
  EXPECT_TRUE(is_leaf(leaf));
}

TEST(IsLeaf, TrueForTreeLeafPtr) {
  TreeNode::Ptr const leaf = TreeLeaf::make(0);
  EXPECT_TRUE(is_leaf(leaf));
}
