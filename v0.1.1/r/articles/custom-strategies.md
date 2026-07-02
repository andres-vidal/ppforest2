# Custom strategies

ppforest2 trains trees by composing six pluggable strategies:

| Strategy                       | Purpose                                            | Built-in                                                                                                                                                                                 |
|--------------------------------|----------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **PP** (projection pursuit)    | Find the projection that best separates groups     | [`pp_pda()`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/pp_pda.md) — Penalized Discriminant Analysis                                                                    |
| **Vars** (variable selection)  | Select which variables are available at each split | [`vars_uniform()`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/vars_uniform.md), [`vars_all()`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/vars_all.md) |
| **Threshold** (split cutpoint) | Compute the split cutpoint in projected space      | [`cutpoint_mean_of_means()`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/cutpoint_mean_of_means.md)                                                                      |
| **Stop** (stopping rule)       | Decide when to stop growing                        | [`stop_pure_node()`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/stop_pure_node.md)                                                                                      |
| **Binarize** (binarization)    | Reduce multiclass to binary at each node           | [`binarize_largest_gap()`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/binarize_largest_gap.md)                                                                          |
| **Grouping** (group partition) | Route observations to children                     | [`grouping_by_label()`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/grouping_by_label.md)                                                                                |

You can add new strategies without modifying the core tree-building
logic. This vignette walks through the process.

## How strategies work

Each strategy is an R list with a `name` field that identifies it, a
`display_name` for summaries, and any parameters the strategy needs. The
`name` must match a C++ strategy registered under the same name.

``` r
library(ppforest2)

pp_pda(0.5)
#> $name
#> [1] "pda"
#> 
#> $display_name
#> [1] "PDA"
#> 
#> $lambda
#> [1] 0.5
#> 
#> attr(,"class")
#> [1] "pp_strategy"
vars_uniform(n_vars = 2)
#> $name
#> [1] "uniform"
#> 
#> $display_name
#> [1] "Uniform random"
#> 
#> $count
#> [1] 2
#> 
#> $p_vars
#> NULL
#> 
#> attr(,"class")
#> [1] "vars_strategy"
cutpoint_mean_of_means()
#> $name
#> [1] "mean_of_means"
#> 
#> $display_name
#> [1] "Mean of means"
#> 
#> attr(,"class")
#> [1] "cutpoint_strategy"
```

When you call
[`pptr()`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/pptr.md)
or
[`pprf()`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/pprf.md),
the strategy lists are passed to C++, where the `name` field dispatches
to the corresponding C++ implementation. The actual computation
(optimization, variable selection, cutpoint) happens entirely in C++.

## Adding a new strategy

Adding a strategy requires work on both sides:

1.  **C++**: Implement the strategy class (the computation).
2.  **R**: Write a constructor function (the user-facing API).

### Step 1: C++ implementation

Each strategy family has a base class with pure virtual methods. Your
new strategy inherits from the appropriate base and implements them.

For example, a new projection pursuit strategy needs to implement
[`optimize()`](https://rdrr.io/r/stats/optimize.html) (find the best
projection):

``` cpp
// File: core/src/models/strategies/pp/MyMethod.hpp
#pragma once
#include "models/strategies/pp/ProjectionPursuit.hpp"
#include "models/strategies/Strategy.hpp"
#include "utils/JsonValidation.hpp"

namespace ppforest2::pp {

struct MyMethod : public ProjectionPursuit {
  explicit MyMethod(float alpha) : alpha_(alpha) {}

  std::string display_name() const override { return "My method"; }

  Result optimize(
    const types::FeatureMatrix&  x,
    const stats::GroupPartition& group_spec) const override {
    // Find the optimal projector for the data.
    // Return Result{ projector_vector, index_value }.
    ...
  }

  nlohmann::json to_json() const override {
    return {{"name", "my_method"}, {"alpha", alpha_}};
  }

  static ProjectionPursuit::Ptr from_json(const nlohmann::json& j) {
    JsonReader{j, "my_method"}.only_keys({"name", "alpha"});
    return my_method(j.at("alpha").get<float>());
  }

  PPFOREST2_REGISTER_STRATEGY(ProjectionPursuit, "my_method")

private:
  const float alpha_;
};

inline ProjectionPursuit::Ptr my_method(float alpha) {
  return std::make_shared<MyMethod>(alpha);
}

}  // namespace ppforest2::pp
```

The key pieces:

- **`to_json()`** serializes the strategy name and parameters. This is
  used for model persistence.
- **`from_json()`** deserializes from JSON and validates that no
  unexpected keys are present.
- **`PPFOREST2_REGISTER_STRATEGY`** registers the factory so JSON
  deserialization finds it automatically.
- **`display_name()`** returns a human-readable label for summaries.
- **Factory function** (`my_method()`) is a convenience wrapper.

The same pattern applies to variable selection strategies (`select()`),
cutpoint strategies (`cutpoint()`), and the other strategy families. See
the C++ documentation for complete interface definitions and examples.

After writing the `.cpp` file, add it to
`core/src/models/CMakeLists.txt`.

### Step 2: R constructor

Write an R function that creates a strategy list. The `name` field must
match the C++ registration name exactly.

``` r
#' My custom projection pursuit strategy.
#'
#' @param alpha A tuning parameter.
#' @return A \code{pp_strategy} object.
#' @export
pp_my_method <- function(alpha = 1.0) {
  if (!is.numeric(alpha) || length(alpha) != 1)
    stop("`alpha` must be a single number.")

  structure(
    list(name = "my_method", display_name = "My method", alpha = alpha),
    class = "pp_strategy"
  )
}
```

The constructor should:

- **Validate parameters** before they reach C++. Catching errors early
  with clear messages is better than a C++ exception.
- **Set the S3 class** to `pp_strategy`, `vars_strategy`,
  `cutpoint_strategy`, `stop_strategy`, `binarize_strategy`, or
  `grouping_strategy`. This is checked by `resolve_strategies()`.
- **Include `display_name`** for readable output in
  [`summary()`](https://rdrr.io/r/base/summary.html).
- **Use the same parameter names** as `to_json()` in C++. The R list is
  converted to JSON and passed to `from_json()` on the C++ side.

### Step 3: Use it

Once both sides are in place, the new strategy works like any built-in:

``` r
# Single tree
tree <- pptr(Species ~ ., data = iris, pp = pp_my_method(alpha = 0.5))

# Forest
forest <- pprf(Species ~ ., data = iris, pp = pp_my_method(alpha = 0.5), vars = vars_uniform(n_vars = 2))

# Summary shows the strategy
summary(tree)
```

The strategy is also available from the CLI:

``` bash
ppforest2 train -d iris.csv --pp my_method:alpha=0.5
```

And models trained with the new strategy can be saved and loaded as
usual — the JSON registry handles serialization automatically.

## Strategy families reference

### PP: Projection pursuit

Controls how the tree finds the best linear combination of variables at
each node.

    index(x, group_spec, projector) -> scalar
    optimize(x, group_spec) -> Result{projector, index}

[`optimize()`](https://rdrr.io/r/stats/optimize.html) is the main
method. It receives the data matrix and group partition and returns the
best projection vector. `index()` evaluates a given projection (used for
variable importance calculations).

### Vars: Variable selection

Controls which variables are available to projection pursuit at each
split. This is what makes random forests “random”.

    select(x, group_spec, rng) -> Result{selected_indices, original_cols}

The returned `VariableSelection::Result` tracks which columns were
selected so the reduced-space projector can be expanded back to the full
feature space.

### Threshold: Split cutpoint

Controls where the split cutpoint is placed in the projected space.

    cutpoint(group_1, group_2, projector) -> scalar

Receives the two groups (already partitioned by projection pursuit) and
the projection vector. Returns the cutpoint value.

### Stop: Stopping rule

Controls when to stop growing the tree.

    should_stop(group_partition, depth) -> bool

### Binarize: Binarization

Controls how multiclass nodes (\>2 groups) are reduced to a binary
problem.

    regroup(projected_x, group_partition) -> Result

### Grouping: Group partition

Controls how observations are routed to children after a split.

    split(partition, lower_group, upper_group) -> Result

## Checklist

1.  Create `core/src/models/strategies/<family>/MyStrategy.hpp` (and
    `.cpp` if needed).
2.  Inherit from the appropriate base class (`ProjectionPursuit`,
    `VariableSelection`, `Cutpoint`, `StopRule`, `Binarization`, or
    `Grouping`).
3.  Implement the pure virtual methods.
4.  Implement `to_json()` with a `"name"` field.
5.  Implement `display_name()` for human-readable summaries.
6.  Add `static Ptr from_json()` with key validation.
7.  Add `PPFOREST2_REGISTER_STRATEGY(Base, "name")`.
8.  Add a factory function in the strategy’s namespace.
9.  Add the `.cpp` to `core/src/models/CMakeLists.txt`.
10. Write tests in `MyStrategy.test.cpp` (JSON round-trip + functional).
11. Write the R constructor function with validation, `display_name`,
    and the correct S3 class.
12. Export and document the R function.
