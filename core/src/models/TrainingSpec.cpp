#include "models/TrainingSpec.hpp"

#include "utils/Invariant.hpp"
#include "utils/UserError.hpp"

#include <string>

namespace ppforest2 {
  // -- Builder -------------------------------------------------------------
  // Default selection and finalize logic out-of-line so `TrainingSpec.hpp`
  // stays on base-class strategy headers and skips the concrete-strategy
  // includes needed only to make factory calls below.

  namespace {
    using BuilderConfig = TrainingSpec::Builder::Config;

    template<types::Mode M> struct ModeTag {};

    template<typename F> auto mode_tag(types::Mode mode, F&& f) {
      switch (mode) {
        case types::Mode::Classification: return f(ModeTag<types::Mode::Classification>{});
        case types::Mode::Regression: return f(ModeTag<types::Mode::Regression>{});
      }
      invariant(false, "mode_tag: unhandled types::Mode value");
    }

    void apply_mode_agnostic_defaults(BuilderConfig& c) {
      if (!c.pp) {
        c.pp = pp::pda(0.0F);
      }
      if (!c.vars) {
        c.vars = vars::all();
      }
      if (!c.cutpoint) {
        c.cutpoint = cutpoint::mean_of_means();
      }
    }

    void apply_mode_defaults(BuilderConfig& c, ModeTag<types::Mode::Classification> /*unused*/) {
      if (!c.stop) {
        c.stop = stop::pure_node();
      }
      if (!c.binarization) {
        c.binarization = binarize::largest_gap();
      }
      if (!c.grouping) {
        c.grouping = grouping::by_label();
      }
      if (!c.leaf) {
        c.leaf = leaf::majority_vote();
      }
    }

    void apply_mode_defaults(BuilderConfig& c, ModeTag<types::Mode::Regression> /*unused*/) {
      if (!c.stop) {
        // `by_cutpoint` grouping benefits from the size/variance pair:
        // bail out early once a node is too small or too pure to split.
        c.stop = stop::any({stop::min_size(5), stop::min_variance(0.01F)});
      }
      if (!c.binarization) {
        // Regression's `by_cutpoint` grouping always yields a 2-group
        // partition, so binarize is a no-op (`disabled`).
        c.binarization = binarize::disabled();
      }
      if (!c.grouping) {
        c.grouping = grouping::by_cutpoint();
      }
      if (!c.leaf) {
        c.leaf = leaf::mean_response();
      }
    }
  }

  TrainingSpec::Builder& TrainingSpec::Builder::apply_defaults() {
    apply_mode_agnostic_defaults(config);
    mode_tag(mode, [&](auto tag) { apply_mode_defaults(config, tag); });
    return *this;
  }

  TrainingSpec TrainingSpec::Builder::build() {
    apply_defaults();
    return TrainingSpec(
        std::move(config.pp),
        std::move(config.vars),
        std::move(config.cutpoint),
        std::move(config.stop),
        std::move(config.binarization),
        std::move(config.grouping),
        std::move(config.leaf),
        mode,
        config.size,
        config.seed,
        config.threads,
        config.max_retries
    );
  }

  TrainingSpec::Ptr TrainingSpec::Builder::make() {
    return std::make_shared<TrainingSpec>(build());
  }

  namespace {
    /**
     * @brief Error if @p strategy does not support @p mode.
     *
     * Uses `user_error` because the failure is driven by user-assembled
     * input (CLI flags, config JSON, R strategy wrappers). The mismatch
     * surfaces as a clean error message rather than a stack-traced
     * internal exception.
     */
    template<typename StrategyPtr>
    void check_supports(StrategyPtr const& strategy, std::string const& family, types::Mode mode) {
      if (!strategy) {
        return;
      }

      auto const modes     = strategy->supported_modes();
      auto const supported = modes.find(mode) != modes.end();
      auto const name      = strategy->display_name();

      user_error(supported, family + " strategy '" + name + "' does not support " + types::to_string(mode) + " mode.");
    }
  }

  TrainingSpec::TrainingSpec(
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
  )
      : pp(std::move(pp))
      , vars(std::move(vars))
      , cutpoint(std::move(cutpoint))
      , stop(std::move(stop))
      , binarization(std::move(binarization))
      , grouping(std::move(grouping))
      , leaf(std::move(leaf))
      , mode(mode)
      , size(size)
      , seed(seed)
      , threads(threads)
      , max_retries(max_retries) {
    // Mode-strategy compatibility check. Fails fast at spec construction.
    check_supports(this->pp, "pp", mode);
    check_supports(this->vars, "vars", mode);
    check_supports(this->cutpoint, "cutpoint", mode);
    check_supports(this->stop, "stop", mode);
    check_supports(this->binarization, "binarize", mode);
    check_supports(this->grouping, "grouping", mode);
    check_supports(this->leaf, "leaf", mode);
  }


  nlohmann::json TrainingSpec::to_json() const {
    nlohmann::json j = {
        {"pp", pp->to_json()},
        {"vars", vars->to_json()},
        {"cutpoint", cutpoint->to_json()},
        {"stop", stop->to_json()},
        {"binarize", binarization->to_json()},
        {"grouping", grouping->to_json()},
        {"leaf", leaf->to_json()},
        {"mode", types::to_string(mode)},
        {"size", size},
        {"seed", seed},
        {"threads", threads},
        {"max_retries", max_retries},
    };

    return j;
  }

  TrainingSpec::Ptr TrainingSpec::from_json(nlohmann::json const& j) {
    types::Mode const mode = types::mode_from_string(j.value("mode", "classification"));

    // Strategy keys are optional. Anything the caller omits is filled in
    // by `Builder::apply_defaults()` at `make()` time, so callers
    // (CLI, R bindings, tests) can send only what the user specified.
    auto b = builder(mode)
                 .size(j.value("size", 0))
                 .seed(j.value("seed", 0))
                 .threads(j.value("threads", 0))
                 .max_retries(j.value("max_retries", 3));

    if (j.contains("pp")) {
      b.pp(pp::ProjectionPursuit::from_json(j.at("pp")));
    }
    if (j.contains("vars")) {
      b.vars(vars::VariableSelection::from_json(j.at("vars")));
    }
    if (j.contains("cutpoint")) {
      b.cutpoint(cutpoint::Cutpoint::from_json(j.at("cutpoint")));
    }
    if (j.contains("stop")) {
      b.stop(stop::StopRule::from_json(j.at("stop")));
    }
    if (j.contains("binarize")) {
      b.binarization(binarize::Binarization::from_json(j.at("binarize")));
    }
    if (j.contains("grouping")) {
      b.grouping(grouping::Grouping::from_json(j.at("grouping")));
    }
    if (j.contains("leaf")) {
      b.leaf(leaf::LeafStrategy::from_json(j.at("leaf")));
    }
    return b.make();
  }

  bool TrainingSpec::should_stop(NodeContext const& ctx, stats::RNG& rng) const {
    return stop->should_stop(ctx, rng);
  }

  void TrainingSpec::find_projection(NodeContext& ctx, stats::RNG& rng) const {
    pp->optimize(ctx, rng);
  }

  void TrainingSpec::select_vars(NodeContext& ctx, stats::RNG& rng) const {
    vars->select(ctx, rng);
  }

  void TrainingSpec::find_cutpoint(NodeContext& ctx, stats::RNG& rng) const {
    cutpoint->locate(ctx, rng);
  }

  void TrainingSpec::regroup(NodeContext& ctx, stats::RNG& rng) const {
    binarization->regroup(ctx, rng);
  }

  void TrainingSpec::group(NodeContext& ctx, stats::RNG& rng) const {
    grouping->split(ctx, rng);
  }
}
