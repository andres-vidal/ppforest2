#pragma once

#include "models/strategies/pp/ProjectionPursuit.hpp"
#include "models/strategies/vars/VariableSelection.hpp"
#include "models/strategies/cutpoint/Cutpoint.hpp"
#include "models/strategies/stop/StopRule.hpp"
#include "models/strategies/binarize/Binarization.hpp"
#include "models/strategies/grouping/Grouping.hpp"
#include "models/strategies/leaf/LeafStrategy.hpp"

#include <memory>
#include <thread>
#include <nlohmann/json.hpp>

namespace ppforest2 {
  /**
   * @brief Training configuration for projection pursuit trees and forests.
   *
   * Composes seven strategies (projection pursuit, variable selection,
   * split cutpoint, stop rule, binarization, grouping, leaf) together with
   * forest-level parameters (size, seed, threads, max retries).
   *
   * TrainingSpec is a concrete class — new strategies are plugged in
   * via the builder, not by subclassing:
   * @code
   *   // Single tree with PDA (lambda = 0.5):
   *   auto spec = TrainingSpec::builder(types::Mode::Classification)
   *       .pp(pp::pda(0.5))
   *       .build();
   *
   *   // Random forest with uniform variable selection:
   *   auto spec = TrainingSpec::builder(types::Mode::Classification)
   *       .size(100)
   *       .pp(pp::pda(0.0))
   *       .vars(vars::uniform(3))
   *       .build();
   * @endcode
   *
   * Strategies are held via shared_ptr and are immutable after
   * construction, so TrainingSpec can be freely copied and shared
   * across trees without deep cloning.
   */
  class TrainingSpec {
  public:
    using Ptr = std::shared_ptr<TrainingSpec>;

    /** @brief Projection pursuit optimization strategy. */
    pp::ProjectionPursuit::Ptr const pp;
    /** @brief Variable selection strategy. */
    vars::VariableSelection::Ptr const vars;
    /** @brief Split cutpoint strategy. */
    cutpoint::Cutpoint::Ptr const cutpoint;
    /** @brief Stop rule strategy. */
    stop::StopRule::Ptr const stop;
    /** @brief Binarization strategy. */
    binarize::Binarization::Ptr const binarization;
    /** @brief Grouping strategy. */
    grouping::Grouping::Ptr const grouping;
    /** @brief Leaf creation strategy. */
    leaf::LeafStrategy::Ptr const leaf;

    /** @brief Training mode (classification or regression). */
    types::Mode const mode;

    /** @brief Number of trees (0 = single tree). */
    int const size;
    /** @brief RNG seed. */
    int const seed;
    /** @brief Number of threads for parallel forest training. */
    int const threads;
    /** @brief Maximum retry attempts for degenerate trees. */
    int const max_retries;

    /**
     * @brief Fluent builder for TrainingSpec.
     *
     * `mode` is required at construction — it's structurally primary
     * (drives the mode-dependent default strategies and gates
     * mode-compatibility checks), not a post-construction tweak, so
     * there's no `.mode()` setter.
     *
     * The Builder's mutable state is grouped in the nested `Config`
     * struct, exposed as the public `config` member. The fluent
     * setters (`.pp(v)`, `.size(5)`, …) keep their conventional names
     * by writing through `config.pp` / `config.size`, so the setter
     * names don't collide with the field names.
     *
     * Strategy fields in `Config` start as `nullptr` and are resolved
     * to mode-aware defaults by `apply_defaults()` — which `build()`
     * invokes automatically. Calling `apply_defaults()` directly is
     * useful when a caller wants to inspect the resolved spec before
     * constructing (e.g. for JSON round-trip tests).
     *
     * The single source of truth for default selection lives in
     * `apply_defaults()` — CLI and R both pass only the strategies the
     * user supplied and rely on the Builder for the rest, so adding a
     * new mode-aware default means changing exactly one location.
     *
     * Strategy factory calls are made in `TrainingSpec.cpp` so the
     * header includes only base-class strategy headers.
     */
    class Builder {
    public:
      /**
       * @brief Builder state — the configuration being assembled.
       *
       * Lives as a nested struct rather than directly on the Builder so
       * the field names (`pp`, `vars`, …) don't collide with the
       * fluent-setter method names of the same names.
       */
      struct Config {
        pp::ProjectionPursuit::Ptr pp            = nullptr;
        vars::VariableSelection::Ptr vars        = nullptr;
        cutpoint::Cutpoint::Ptr cutpoint         = nullptr;
        stop::StopRule::Ptr stop                 = nullptr;
        binarize::Binarization::Ptr binarization = nullptr;
        grouping::Grouping::Ptr grouping         = nullptr;
        leaf::LeafStrategy::Ptr leaf             = nullptr;

        int size        = 0;
        int seed        = 0;
        int threads     = 0;
        int max_retries = 3;
      };

      Config config;
      types::Mode const mode;

      explicit Builder(types::Mode mode)
          : mode(mode) {}

      Builder& pp(pp::ProjectionPursuit::Ptr v) {
        config.pp = std::move(v);
        return *this;
      }
      Builder& vars(vars::VariableSelection::Ptr v) {
        config.vars = std::move(v);
        return *this;
      }
      Builder& cutpoint(cutpoint::Cutpoint::Ptr v) {
        config.cutpoint = std::move(v);
        return *this;
      }
      Builder& stop(stop::StopRule::Ptr v) {
        config.stop = std::move(v);
        return *this;
      }
      Builder& binarization(binarize::Binarization::Ptr v) {
        config.binarization = std::move(v);
        return *this;
      }
      Builder& grouping(grouping::Grouping::Ptr v) {
        config.grouping = std::move(v);
        return *this;
      }
      Builder& leaf(leaf::LeafStrategy::Ptr v) {
        config.leaf = std::move(v);
        return *this;
      }

      Builder& size(int v) {
        config.size = v;
        return *this;
      }
      Builder& seed(int v) {
        config.seed = v;
        return *this;
      }
      Builder& threads(int v) {
        config.threads = v;
        return *this;
      }
      Builder& max_retries(int v) {
        config.max_retries = v;
        return *this;
      }

      /**
       * @brief Fill in any null strategy fields with mode-aware defaults.
       *
       * Idempotent — only assigns when a field is currently `nullptr`,
       * so re-calling after explicit overrides leaves user choices intact.
       *
       * Dispatches on `mode` exactly once (via the per-mode
       * `apply_mode_defaults` overloads in the implementation file), so
       * adding a new mode means adding a new overload — not adding more
       * `is_regression` branches throughout the function.
       *
       * Defaults by mode:
       * - `pp`:       `pp::pda(0.0)`                 (both modes)
       * - `vars`:     `vars::all()`                  (both modes)
       * - `cutpoint`: `cutpoint::mean_of_means()`    (both modes)
       * - `stop`:     `stop::pure_node()` for classification;
       *               `stop::any({min_size(5), min_variance(0.01)})` for regression
       * - `binarize`: `binarize::largest_gap()` for classification;
       *               `binarize::disabled()` for regression
       *               (regression's `by_cutpoint` grouping always yields
       *               2 groups, so binarize is a no-op)
       * - `grouping`: `grouping::by_label()` for classification;
       *               `grouping::by_cutpoint()` for regression
       * - `leaf`:     `leaf::majority_vote()` for classification;
       *               `leaf::mean_response()` for regression
       *
       * Returns `*this` for fluent chaining.
       */
      Builder& apply_defaults();

      /**
       * @brief Finalize the builder into a `TrainingSpec`.
       *
       * Calls `apply_defaults()` first so any unset strategies receive
       * their mode-aware default, then constructs and returns the spec.
       */
      TrainingSpec build();

      /** @brief Shorthand for `std::make_shared<TrainingSpec>(build())`. */
      Ptr make();
    };

    /**
     * @brief Create a builder for the given mode.
     *
     * Mode is required because it's structurally primary: it determines
     * the default `binarize` strategy and the mode-compatibility checks
     * for every other strategy. There is no "no-mode" TrainingSpec.
     */
    static Builder builder(types::Mode mode) { return Builder{mode}; }

    /**
     * @brief Construct a training specification.
     *
     * @param pp           Projection pursuit strategy.
     * @param vars         Variable selection strategy.
     * @param cutpoint     Split cutpoint strategy.
     * @param stop         Stop rule strategy.
     * @param binarization Binarization strategy.
     * @param grouping     Grouping strategy.
     * @param leaf         Leaf creation strategy.
     * @param size         Number of trees (0 = single tree).
     * @param seed         RNG seed.
     * @param threads      Number of threads (0 = hardware concurrency).
     * @param max_retries  Maximum retry attempts for degenerate trees.
     */
    TrainingSpec(
        pp::ProjectionPursuit::Ptr pp,
        vars::VariableSelection::Ptr vars,
        cutpoint::Cutpoint::Ptr cutpoint,
        stop::StopRule::Ptr stop,
        binarize::Binarization::Ptr binarization,
        grouping::Grouping::Ptr grouping,
        leaf::LeafStrategy::Ptr leaf,
        types::Mode mode,
        int size,
        int seed,
        int threads,
        int max_retries
    );

    // -- Forwarding methods (delegate to the underlying strategy) -----------

    /** @brief Run projection pursuit optimization. Asserts postcondition: `ctx.projector` and `ctx.pp_index_value` are set. */
    void find_projection(NodeContext& ctx, stats::RNG& rng) const;

    /** @brief Run variable selection. Asserts postcondition: `ctx.var_selection` is set. */
    void select_vars(NodeContext& ctx, stats::RNG& rng) const;

    /** @brief Compute the split cutpoint. Asserts postcondition: `ctx.cutpoint` is set. */
    void find_cutpoint(NodeContext& ctx, stats::RNG& rng) const;

    /**
     * @brief Check whether the node should stop growing.
     *
     * Returns true if the configured stop rule fires, OR if the node has
     * fewer than 2 groups (a tree-level invariant — no configured rule
     * can split a single group, so stopping is the only sane option).
     */
    bool should_stop(NodeContext const& ctx, stats::RNG& rng) const;

    /** @brief Reduce multiclass partition to binary. Asserts postcondition: `ctx.y_bin` is set. */
    void regroup(NodeContext& ctx, stats::RNG& rng) const;

    /**
     * @brief Split observations into two child partitions.
     *
     * Writes `ctx.lower_y_part` / `ctx.upper_y_part`. Sets `ctx.aborted`
     * if the grouping produced no progress (i.e. one child covers the
     * whole parent).
     */
    void group(NodeContext& ctx, stats::RNG& rng) const;

    /** @brief Create the initial group partition from the training response. */
    stats::GroupPartition init_groups(types::OutcomeVector const& y) const { return grouping->init(y); }

    /** @brief Create a leaf node from the current node context. */
    TreeNode::Ptr create_leaf(NodeContext const& ctx, stats::RNG& rng) const { return leaf->create_leaf(ctx, rng); }

    /** @brief Whether this specification describes a forest (size > 0). */
    bool is_forest() const { return size > 0; }

    /** @brief Serialize the training spec to JSON. */
    nlohmann::json to_json() const;

    /** @brief Deserialize a training spec from JSON. */
    static Ptr from_json(nlohmann::json const& j);

    /** @brief Create a shared pointer to a TrainingSpec. */
    template<typename... Args> static Ptr make(Args&&... args) {
      return std::make_shared<TrainingSpec>(std::forward<Args>(args)...);
    }

    /** @brief
     * Get the number of threads to use for training.
     *
     * If the number of threads is not specified, the number of hardware
     * concurrency is returned.
     *
     * @return The number of threads to use for training.
     */
    int resolve_threads() const {
      return threads > 0 ? threads : static_cast<int>(std::thread::hardware_concurrency());
    }
  };

  /** @brief Whether @p spec describes a classification training run. */
  inline bool is_classification(TrainingSpec const& spec) {
    return types::is_classification(spec.mode);
  }

  /** @brief Whether @p spec describes a regression training run. */
  inline bool is_regression(TrainingSpec const& spec) {
    return types::is_regression(spec.mode);
  }

  /** @brief Pointer overload — null-safe; returns `false` for a null spec. */
  inline bool is_classification(TrainingSpec::Ptr const& spec) {
    return spec != nullptr && is_classification(*spec);
  }

  /** @brief Pointer overload — null-safe; returns `false` for a null spec. */
  inline bool is_regression(TrainingSpec::Ptr const& spec) {
    return spec != nullptr && is_regression(*spec);
  }
}
