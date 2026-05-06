#pragma once

#include "models/strategies/pp/ProjectionPursuit.hpp"
#include "models/strategies/Strategy.hpp"
#include "utils/Types.hpp"

namespace ppforest2::pp {
  /**
   * @brief Penalized Discriminant Analysis projection pursuit strategy.
   *
   * Optimizes a linear discriminant projection using a penalized
   * between-group / within-group variance ratio. The @c lambda
   * parameter controls the penalty strength in the LDA index.
   */
  class PDA : public ProjectionPursuit {
  public:
    explicit PDA(float lambda);

    static ProjectionPursuit::Ptr from_json(nlohmann::json const& j);

    nlohmann::json to_json() const override;

    std::string display_name() const override { return lambda == 0 ? "LDA" : "PDA"; }

    std::set<types::Mode> supported_modes() const override {
      return {types::Mode::Classification, types::Mode::Regression};
    }

    PPFOREST2_REGISTER_STRATEGY(ProjectionPursuit, "pda")
    PPFOREST2_REGISTER_PRIMARY_PARAM("pda", "lambda")

  protected:
    void compute(NodeContext& ctx, stats::RNG& rng) const override;

  private:
    /** @brief Penalty parameter for the LDA index (0 = standard LDA). */
    float const lambda;
  };
}
