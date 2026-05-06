#include "models/Forest.hpp"

#include "models/ClassificationForest.hpp"
#include "models/Model.hpp"
#include "models/RegressionForest.hpp"
#include "utils/Invariant.hpp"

#include <cstdint>
#include <exception>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace ppforest2::stats;
using namespace ppforest2::types;

namespace ppforest2 {

  void Forest::build_trees(FeatureMatrix const& x, OutcomeVector const& y) {
    invariant(this->training_spec != nullptr, "Forest::build_trees: must have training spec");
    invariant(training_spec->size > 0, "The forest size must be greater than 0.");

    std::uint64_t const size        = static_cast<std::uint64_t>(training_spec->size);
    std::uint64_t const seed        = static_cast<std::uint64_t>(training_spec->seed);
    std::uint64_t const max_retries = static_cast<std::uint64_t>(training_spec->max_retries);

    // clang-format off
    #ifdef _OPENMP
    omp_set_num_threads(training_spec->resolve_threads());
    #endif
    // clang-format on

    std::vector<BaggedTree::Ptr> bags(size);
    std::vector<std::exception_ptr> errors(size);

    // clang-format off
    #pragma omp parallel for schedule(static)
    // clang-format on
    for (std::uint64_t i = 0; i < size; ++i) {
      for (std::uint64_t attempt = 0; attempt <= max_retries; ++attempt) {
        try {
          // Stream id is load-bearing for golden-file reproducibility (see
          // CLAUDE.md). Don't change without regenerating every golden file.
          std::uint64_t const stream = i + attempt * size;
          RNG rng(seed, stream);

          bags[i] = train_tree(x, y, rng);

          if (!bags[i]->degenerate()) {
            break;
          }
        } catch (...) {
          errors[i] = std::current_exception();
          break;
        }
      }
    }

    // Rethrow the lowest-index exception, if any. Done outside the parallel
    // region so the throw runs on the main thread.
    for (auto const& e : errors) {
      if (e) {
        std::rethrow_exception(e);
      }
    }

    for (auto& bag : bags) {
      add_tree(std::move(bag));
    }
  }

  void Forest::add_tree(BaggedTree::Ptr tree) {
    invariant(this->training_spec != nullptr, "Forest::add_tree: must have training spec");
    invariant(tree != nullptr, "Forest::add_tree: tree is null");
    invariant(tree->model->training_spec != nullptr, "Forest::add_tree: tree must have training spec");
    invariant(
        tree->model->training_spec->mode == this->training_spec->mode,
        "Forest::add_tree: tree mode does not match forest mode "
        "(attempted to add a tree trained under a different mode)"
    );

    // A forest is degenerate iff any of its trees is — propagate here so
    // every code path that attaches a tree (build_trees, JSON load, hand-
    // assembled test fixtures) keeps the flag in sync without duplication.
    if (tree->degenerate()) {
      this->degenerate = true;
    }

    trees.push_back(std::move(tree));
  }

  bool Forest::operator==(Forest const& other) const {
    if (trees.size() != other.trees.size()) {
      return false;
    }

    for (std::size_t i = 0; i < trees.size(); ++i) {
      if (*trees[i] != *other.trees[i]) {
        return false;
      }
    }

    return true;
  }

  bool Forest::operator!=(Forest const& other) const {
    return !(*this == other);
  }

  Forest::Ptr Forest::train(TrainingSpec const& spec, FeatureMatrix const& x, OutcomeVector const& y) {
    Model::check_train_inputs(x, y);

    Forest::Ptr forest;

    if (is_regression(spec)) {
      forest = RegressionForest::train(spec, x, y);
    } else {
      forest = ClassificationForest::train(spec, x, y);
    }

    return forest;
  }

}
