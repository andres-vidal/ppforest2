#pragma once

#include "models/Projector.hpp"
#include "models/strategies/Strategy.hpp"
#include "stats/Stats.hpp"
#include "utils/Types.hpp"

/**
 * @brief Projection pursuit strategies.
 *
 * Contains the abstract ProjectionPursuit interface and concrete
 * implementations (e.g. PDA) that define how to evaluate
 * and optimise a projection index for separating groups.
 */
namespace ppforest2 {
  struct NodeContext;
}

namespace ppforest2::pp {
  /**
   * @brief Abstract strategy for projection pursuit optimization.
   *
   * Writes `ctx.projector` and `ctx.pp_index_value`.
   */
  class ProjectionPursuit : public Strategy<ProjectionPursuit> {
  public:
    /**
     * @brief Result of a projection pursuit optimization step.
     */
    struct Result {
      /** @brief Optimized projection vector. */
      ppforest2::pp::Projector projector;
      /** @brief Projection pursuit index value achieved. */
      types::Feature index_value = 0;
    };

    /** @brief Find the optimal projection and store it in the context. */
    void optimize(NodeContext& ctx, stats::RNG& rng) const;

  protected:
    virtual void compute(NodeContext& ctx, stats::RNG& rng) const = 0;
  };

  /** @brief Factory function for a PDA projection pursuit strategy. */
  ProjectionPursuit::Ptr pda(float lambda);
}
