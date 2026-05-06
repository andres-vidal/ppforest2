#pragma once

#include "utils/Invariant.hpp"
#include "utils/Types.hpp"

#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace ppforest2::strategies {
  /**
   * @brief Cross-family registry for strategies that accept a CLI
   *        positional-shorthand value (e.g. `min_size:5` instead of
   *        `min_size:min_size=5`).
   *
   * Keyed on the strategy's registered name (e.g. "min_size", "pda").
   * Single global map across all families, relying on the codebase's
   * existing convention that strategy names are unique across families
   * (registered via `PPFOREST2_REGISTER_STRATEGY`). Collisions are
   * caught at static-init time and raise a clear error rather than
   * silently overwriting.
   *
   * The registration is co-located with the strategy class in its
   * header (via `PPFOREST2_REGISTER_PRIMARY_PARAM`) so the shorthand
   * metadata travels with the class definition — not as an out-of-
   * band lookup in the parser. Consumers (currently only
   * `cli::strategy_string_to_json`) query via `primary_param_for`.
   */
  inline std::map<std::string, std::string>& primary_params_registry() {
    static std::map<std::string, std::string> r;
    return r;
  }

  /**
   * @brief Register a strategy's primary (shorthand) parameter name.
   *
   * Called via `PPFOREST2_REGISTER_PRIMARY_PARAM` alongside the
   * existing `PPFOREST2_REGISTER_STRATEGY` macro.
   *
   * Fires an `invariant` if @p strategy is already in the registry —
   * a programmer error (two strategies declared shorthand for the
   * same name) that can only originate from source code, not runtime
   * input. Double-registration with identical values is also caught:
   * the `inline static` definition of `registered_primary_param_` in
   * the macro is ODR-merged across TUs to a single initializer, so
   * each strategy reaches this function exactly once per program;
   * any second entry is a name collision, not an idempotent retry.
   * An invariant abort is safer than throwing from a static
   * initializer: the abort prints the reason cleanly, whereas an
   * exception escaping a global constructor can trigger
   * `std::terminate` with confusing diagnostics.
   */
  inline bool register_primary_param(std::string const& strategy, std::string const& param) {
    auto [it, inserted] = primary_params_registry().try_emplace(strategy, param);
    invariant(inserted, "Conflicting primary_param for '" + strategy + "': '" + it->second + "' vs '" + param + "'");
    return true;
  }

  /**
   * @brief Look up a strategy's primary shorthand parameter by name.
   *
   * @return The parameter name if registered; `std::nullopt` otherwise
   *         (strategy not shorthand-eligible, or strategy unknown).
   */
  inline std::optional<std::string> primary_param_for(std::string const& strategy) {
    auto& r = primary_params_registry();
    if (auto it = r.find(strategy); it != r.end()) {
      return it->second;
    }
    return std::nullopt;
  }
}

/**
 * @brief CRTP base class providing self-registration for strategy types.
 *
 * Each strategy family (ProjectionPursuit, VariableSelection, Cutpoint,
 * StopRule, Binarization, Grouping) inherits from Strategy<Derived>
 * to get a shared_ptr Ptr typedef, a static registry of named factory
 * functions, and a from_json() dispatcher that looks up the "name"
 * field and delegates to the registered factory.
 *
 * Concrete strategies register themselves at static-init time via
 * the PPFOREST2_REGISTER_STRATEGY macro.
 *
 * @tparam Derived  The strategy base class (e.g. ProjectionPursuit).
 */
template<typename Derived> class Strategy {
public:
  using Ptr = std::shared_ptr<Derived>;

  /** @brief Factory function type for deserializing a strategy from JSON. */
  using Factory = Ptr (*)(nlohmann::json const&);

  virtual ~Strategy() = default;

  /**
   * @brief Serialize this strategy's configuration to JSON.
   *
   * Must include a "name" field identifying the strategy type,
   * plus any strategy-specific parameters.
   */
  virtual nlohmann::json to_json() const = 0;

  /** @brief Human-readable name for display in summaries. */
  virtual std::string display_name() const = 0;

  /**
   * @brief Training modes this strategy supports.
   *
   * Every concrete strategy MUST explicitly declare which modes it
   * supports — there is no default. This forces implementers to think
   * about mode compatibility rather than silently inheriting both modes
   * and triggering runtime surprises.
   *
   * Used by TrainingSpec to validate that every selected strategy is
   * compatible with the configured training mode. Fails fast at build
   * time instead of silently misbehaving at train time.
   */
  virtual std::set<ppforest2::types::Mode> supported_modes() const = 0;

  /**
   * @brief Register a concrete strategy for JSON deserialization.
   *
   * Called automatically by concrete strategies via inline static
   * initialization.  Adding a new strategy only requires defining
   * a static `from_json` and the PPFOREST2_REGISTER_STRATEGY macro.
   *
   * @param name     Strategy name (must match the "name" field in JSON).
   * @param factory  Factory function that constructs the strategy.
   * @return         Always true (return value used for static init).
   */
  static bool register_strategy(std::string const& name, Factory factory) {
    registry()[name] = factory;
    return true;
  }

  /**
   * @brief Construct a strategy from its JSON representation.
   *
   * Dispatches on the "name" field to the registered factory.
   *
   * @param j  JSON object (must contain "name").
   * @return   Shared pointer to the constructed strategy.
   */
  static Ptr from_json(nlohmann::json const& j) {
    auto name = j.at("name").get<std::string>();
    auto& reg = registry();
    auto it   = reg.find(name);

    if (it == reg.end()) {
      throw std::runtime_error("Unknown strategy: " + name);
    }

    return it->second(j);
  }

private:
  static std::map<std::string, Factory>& registry() {
    static std::map<std::string, Factory> r;
    return r;
  }
};

/**
 * @brief Auto-registration macro for strategy factories.
 *
 * Registers `ConcreteStrategy::from_json` as the factory for @p name
 * in the given @p StrategyBase class.
 *
 * Usage (inside the concrete strategy class, after from_json):
 * @code
 *   class PDA : public ProjectionPursuit {
 *    public:
 *     static ProjectionPursuit::Ptr from_json(const nlohmann::json& j);
 *     PPFOREST2_REGISTER_STRATEGY(ProjectionPursuit, "pda")
 *   };
 * @endcode
 */
#define PPFOREST2_REGISTER_STRATEGY(StrategyBase, name) \
  inline static const bool registered_ = StrategyBase::register_strategy(name, from_json);


/**
 * @brief Declare the strategy's CLI positional-shorthand parameter.
 *
 * Optional companion to `PPFOREST2_REGISTER_STRATEGY`. When present,
 * users can write `--flag <name>:<value>` instead of
 * `--flag <name>:<primary_param>=<value>` on the CLI.
 *
 * Usage (after `PPFOREST2_REGISTER_STRATEGY`, inside the class body):
 * @code
 *   class MinSize : public StopRule {
 *    public:
 *     static StopRule::Ptr from_json(const nlohmann::json& j);
 *     PPFOREST2_REGISTER_STRATEGY(StopRule, "min_size")
 *     PPFOREST2_REGISTER_PRIMARY_PARAM("min_size", "min_size")
 *   };
 * @endcode
 *
 * Takes the strategy name as the first argument (redundant with
 * `PPFOREST2_REGISTER_STRATEGY`, but the macro can't see that one's
 * arguments) and the shorthand-parameter name as the second.
 *
 * Only numeric primary parameters are exercised by the parser today —
 * see `cli::strategy_string_to_json`. String-valued primary params
 * would conflict with bare identifiers (e.g. `pda:lambda` is a typo,
 * not a value), so shorthand gates on the post-`:` token being
 * numeric. If a future strategy needs a string primary param, the
 * parser needs a matching update.
 *
 * **Placement matters**: put this macro inside the class body next to
 * `PPFOREST2_REGISTER_STRATEGY`, not in the `.cpp`. Both macros expand
 * to `inline static` members whose initializers fire at C++ global-
 * init time. Co-located in the same header, they pull in together:
 * the force-link block in `TrainingSpec.cpp` keeps the strategy's
 * `.cpp.o` alive, which transitively keeps the header's inline
 * statics alive as a unit. Split them across `.hpp` / `.cpp` and
 * aggressive dead-stripping could drop one but not the other —
 * symptom is "shorthand doesn't work for this strategy" with no
 * other signal.
 */
#define PPFOREST2_REGISTER_PRIMARY_PARAM(strategy_name, param_name) \
  inline static const bool registered_primary_param_ =              \
      ppforest2::strategies::register_primary_param(strategy_name, param_name);
