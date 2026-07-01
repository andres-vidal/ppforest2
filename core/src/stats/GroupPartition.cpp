#include "stats/GroupPartition.hpp"

#include "stats/Stats.hpp"
#include "utils/Invariant.hpp"
#include "utils/Map.hpp"

#include <stdexcept>

using namespace ppforest2::types;

namespace ppforest2::stats {
  bool GroupPartition::is_contiguous(GroupVector const& y) {
    GroupSet visited;

    for (int i = 0; i < y.rows(); i++) {
      if (visited.count(y(i)) == 0) {
        visited.insert(y(i));
      } else if (y(i - 1) != y(i)) {
        return false;
      }
    }

    return true;
  }

  GroupPartition::BlockMap GroupPartition::init_blocks(GroupVector const& y) {
    std::map<Group, Block> blocks;

    for (int i = 0; i < y.rows(); i++) {
      if (blocks.count(y(i)) == 0) {
        Group curr         = y(i);
        blocks[curr].start = i;

        if (i != 0) {
          Group prev        = y(i - 1);
          blocks[curr].prev = prev;
          blocks[prev].next = curr;
          blocks[prev].end  = i - 1;
          blocks[prev].size = i - blocks[prev].start;
        }
      } else if (y(i - 1) != y(i)) {
        throw std::invalid_argument("GroupPartition: data is not organized in contiguous groups");
      }
    }

    Group const last     = y(y.rows() - 1);
    blocks.at(last).end  = y.rows() - 1;
    blocks.at(last).size = y.rows() - blocks.at(last).start;

    return blocks;
  }

  GroupPartition::GroupMap GroupPartition::init_supergroups(GroupSet const& groups) {
    GroupMap sg;

    for (Group const& g : groups) {
      sg[g] = g;
    }

    return sg;
  }

  GroupPartition::GroupPartition(GroupVector const& y)
      : groups(unique(y))
      , supergroups(init_supergroups(groups))
      , subgroups(utils::invert(supergroups))
      , blocks(init_blocks(y)) {}

  GroupPartition::GroupPartition(types::OutcomeVector const& y)
      : GroupPartition(GroupIdVector(y.cast<GroupId>())) {}

  GroupPartition::GroupPartition(BlockMap const& blocks)
      : groups(utils::keys(blocks))
      , supergroups(init_supergroups(groups))
      , subgroups(utils::invert(supergroups))
      , blocks(blocks) {}

  GroupPartition::GroupPartition(BlockMap const& blocks, GroupSet const& groups_)
      : groups(groups_)
      , supergroups(init_supergroups(groups_))
      , subgroups(utils::invert(supergroups))
      , blocks(blocks) {}

  GroupPartition::GroupPartition(BlockMap const& blocks, GroupMap const& supergroups_)
      : groups(utils::values(supergroups_))
      , supergroups(supergroups_)
      , subgroups(utils::invert(supergroups))
      , blocks(blocks) {}

  GroupPartition::GroupPartition(BlockMap const& blocks, GroupSet const& groups_, GroupMap const& supergroups_)
      : groups(groups_)
      , supergroups(supergroups_)
      , subgroups(utils::invert(supergroups))
      , blocks(blocks) {}

  int GroupPartition::group_start(Group const& group) const {
    return blocks.at(group).start;
  }

  int GroupPartition::group_end(Group const& group) const {
    return blocks.at(group).end;
  }

  int GroupPartition::group_size(Group const& group) const {
    return blocks.at(group).size;
  }

  int GroupPartition::total_size() const {
    int total = 0;
    for (auto const& kv : blocks) {
      total += kv.second.size;
    }
    return total;
  }

  FeatureVector GroupPartition::mean(FeatureMatrix const& x) const {
    return data(x).colwise().mean();
  }

  FeatureMatrix GroupPartition::bgss(FeatureMatrix const& x) const {
    FeatureVector const global_mean = mean(x);
    FeatureMatrix result            = FeatureMatrix::Zero(x.cols(), x.cols());

    GroupSet const active_groups = utils::values(supergroups);

    for (Group const& g : active_groups) {
      auto group_data = group(x, g);
      auto group_mean = group_data.colwise().mean().transpose();
      auto centered   = group_mean - global_mean;

      result += group_data.rows() * (centered * centered.transpose());
    }

    return result;
  }

  FeatureMatrix GroupPartition::wgss(FeatureMatrix const& x) const {
    FeatureMatrix result = FeatureMatrix::Zero(x.cols(), x.cols());

    GroupSet const active_groups = utils::values(supergroups);

    for (Group const& g : active_groups) {
      auto group_data = group(x, g);
      auto centered   = group_data.rowwise() - group_data.colwise().mean();

      result += centered.transpose() * centered;
    }

    return result;
  }

  GroupPartition GroupPartition::subset(GroupSet const& groups) const {
    BlockMap subset_blocks;
    std::optional<Group> prev;

    for (auto const& g : groups) {
      Block block = {blocks.at(g).start, blocks.at(g).end, blocks.at(g).size};

      if (prev) {
        block.prev                   = prev;
        subset_blocks.at(*prev).next = g;
      }

      subset_blocks[g] = block;
      prev             = g;
    }

    return GroupPartition(subset_blocks, groups);
  }

  std::pair<GroupPartition, GroupPartition> GroupPartition::split(SplitSizes const& left_sizes) const {
    BlockMap l_blocks;
    BlockMap r_blocks;

    for (auto const& [g, block] : blocks) {
      int const left_count = left_sizes.count(g) != 0U ? left_sizes.at(g) : 0;

      invariant(left_count >= 0 && left_count <= block.size, "GroupPartition::split: left_count out of range");

      if (left_count > 0) {
        l_blocks.emplace(g, Block{block.start, block.start + left_count - 1, left_count});
      }

      if (left_count < block.size) {
        r_blocks.emplace(g, Block{block.start + left_count, block.end, block.size - left_count});
      }
    }

    return {GroupPartition(l_blocks), GroupPartition(r_blocks)};
  }

  GroupPartition::GroupPartition(int start, int end)
      : GroupPartition([&] {
        invariant(start >= 0 && end >= start, "GroupPartition(start, end): invalid range");
        BlockMap blocks;
        blocks[0] = Block{start, end, end - start + 1, std::nullopt, std::nullopt};
        return blocks;
      }()) {}

  GroupPartition GroupPartition::bisect(int mid) const {
    invariant(blocks.size() == 1, "GroupPartition::bisect: requires a single-group partition");
    auto const& only = blocks.begin()->second;
    int const start  = only.start;
    int const end    = only.end;
    invariant(start < mid && mid <= end, "GroupPartition::bisect: mid must be in (start, end]");

    BlockMap split_blocks;
    split_blocks[0] = Block{start, mid - 1, mid - start, 1, std::nullopt};
    split_blocks[1] = Block{mid, end, end - mid + 1, std::nullopt, 0};

    return GroupPartition(split_blocks);
  }

  GroupPartition GroupPartition::remap(GroupMap const& mapping) const {
    return GroupPartition(blocks, mapping);
  }

  GroupPartition GroupPartition::collapse() const {
    GroupMap mapping;
    for (auto const& g : groups) {
      mapping[g] = 0;
    }

    return remap(mapping);
  }
}
