#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "models/strategies/leaf/MajorityVote.hpp"
#include "models/strategies/leaf/LeafStrategy.hpp"
#include "test/NodeContextFixture.hpp"
#include "utils/Types.hpp"
#include "utils/Macros.hpp"

using namespace ppforest2;
using namespace ppforest2::leaf;
using namespace ppforest2::stats;
using namespace ppforest2::test;
using namespace ppforest2::types;
using json = nlohmann::json;

TEST(MajorityVoteLeaf, FromJsonValid) {
  json const j  = {{"name", "majority_vote"}};
  auto strategy = MajorityVote::from_json(j);
  ASSERT_NE(strategy, nullptr);
}

TEST(MajorityVoteLeaf, FromJsonRoundTrip) {
  json const j = {{"name", "majority_vote"}};

  auto strategy = MajorityVote::from_json(j);

  EXPECT_EQ(strategy->to_json(), j);
}

TEST(MajorityVoteLeaf, FromJsonUnknownParam) {
  json const j = {{"name", "majority_vote"}, {"extra", 0}};
  EXPECT_THROW(MajorityVote::from_json(j), std::runtime_error);
}

TEST(MajorityVoteLeaf, RegistryLookup) {
  json const j = {{"name", "majority_vote"}};

  auto strategy = LeafStrategy::from_json(j);

  ASSERT_NE(strategy, nullptr);
  EXPECT_EQ(strategy->to_json(), j);
}

TEST(MajorityVoteLeaf, RegistryUnknownStrategy) {
  json const j = {{"name", "unknown_leaf"}};
  EXPECT_THROW(LeafStrategy::from_json(j), std::runtime_error);
}

TEST(MajorityVoteLeaf, SingleGroupReturnsIt) {
  NodeContextFixture f(MAT(Feature, rows(3), 1, 2, 3, 4, 5, 6), VEC(GroupId, 2, 2, 2));

  MajorityVote const mv;
  auto leaf = mv.create_leaf(f.ctx, f.rng);

  ASSERT_NE(leaf, nullptr);
  EXPECT_TRUE(is_leaf(leaf));
  EXPECT_EQ(leaf->response(), 2);
  EXPECT_FALSE(leaf->degenerate);
}

TEST(MajorityVoteLeaf, MajorityGroupWins) {
  NodeContextFixture f(MAT(Feature, rows(4), 1, 2, 3, 4, 5, 6, 7, 8), VEC(GroupId, 0, 0, 0, 1));

  MajorityVote const mv;
  auto leaf = mv.create_leaf(f.ctx, f.rng);

  EXPECT_EQ(leaf->response(), 0);
  EXPECT_FALSE(leaf->degenerate);
}

TEST(MajorityVoteLeaf, TiedGroupsPicksSmallestLabel) {
  NodeContextFixture f(MAT(Feature, rows(4), 1, 2, 3, 4, 5, 6, 7, 8), VEC(GroupId, 0, 0, 1, 1));

  MajorityVote const mv;
  auto leaf = mv.create_leaf(f.ctx, f.rng);

  // On tie, iterating std::set<int> yields smallest first, which gets
  // majority_size first and subsequent ties don't replace it.
  EXPECT_EQ(leaf->response(), 0);
}

TEST(MajorityVoteLeaf, ThreeGroupsMajorityWins) {
  NodeContextFixture f(MAT(Feature, rows(6), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), VEC(GroupId, 0, 1, 1, 1, 2, 2));

  MajorityVote const mv;
  auto leaf = mv.create_leaf(f.ctx, f.rng);

  EXPECT_EQ(leaf->response(), 1);
}

TEST(MajorityVoteLeaf, DisplayName) {
  MajorityVote const mv;
  EXPECT_EQ(mv.display_name(), "Majority vote");
}
