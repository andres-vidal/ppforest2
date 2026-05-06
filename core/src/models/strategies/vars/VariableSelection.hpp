#pragma once

#include "models/Projector.hpp"
#include "models/strategies/Strategy.hpp"
#include "stats/Stats.hpp"

#include <vector>

/**
 * @brief Variable selection strategies.
 *
 * Contains the abstract VariableSelection interface and concrete
 * implementations that select a subset of variables before projection
 * pursuit optimisation. All uses all variables (single trees);
 * Uniform samples uniformly at random (forests).
 */
namespace ppforest2 {
  struct NodeContext;
}

namespace ppforest2::vars {
  /**
   * @brief Abstract strategy for variable selection.
   *
   * Writes `ctx.var_selection`.
   */
  class VariableSelection : public Strategy<VariableSelection> {
  public:
    /**
     * @brief Result of a variable selection step.
     *
     * Records which columns were selected and allows expanding a
     * reduced-dimension projector back to the original feature space.
     */
    struct Result {
      /** @brief Indices of the selected columns in the original matrix. */
      std::vector<int> selected_cols;
      /** @brief Number of columns in the original (unreduced) matrix. */
      size_t original_size = 0;

      Result() = default;

      Result(std::vector<int> const& selected_cols, size_t original_size)
          : selected_cols(selected_cols)
          , original_size(original_size) {}

      /**
       * @brief Expand a reduced-dimension projector to the original space.
       *
       * Places each element of @p reduced_vector at the position of
       * the corresponding selected column; all other positions are zero.
       *
       * @param reduced_vector  Projector in the reduced space (q).
       * @return                Projector in the original space (p), zero-padded.
       */
      ::ppforest2::pp::Projector expand(::ppforest2::pp::Projector const& reduced_vector) const {
        ::ppforest2::pp::Projector full_vector = ::ppforest2::pp::Projector::Zero(original_size);

        for (size_t i = 0; i < selected_cols.size(); ++i) {
          full_vector(selected_cols[i]) = reduced_vector(i);
        }

        return full_vector;
      }
    };

    /**
     * @brief Select a subset of variables and store the result in the context.
     *
     * Public NVI entry point: skips if `ctx.aborted` is set, otherwise
     * delegates to the subclass-provided `compute`.
     */
    void select(NodeContext& ctx, stats::RNG& rng) const;

  protected:
    /** @brief Subclass implementation of variable selection. */
    virtual void compute(NodeContext& ctx, stats::RNG& rng) const = 0;
  };

  /** @brief Factory function: select all variables (no selection). */
  VariableSelection::Ptr all();

  /** @brief Factory function: uniform random variable selection. */
  VariableSelection::Ptr uniform(int n_vars);
}
