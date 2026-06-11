# Parsnip model specification for pprf.

Creates a model specification for a Projection Pursuit random forest.
Use `set_engine("ppforest2")` to select the ppforest2 engine.

## Usage

``` r
pp_rand_forest(
  mode = "classification",
  trees = NULL,
  mtry = NULL,
  penalty = NULL
)
```

## Arguments

- mode:

  A character string for the model type. Either `"classification"` or
  `"regression"`.

- trees:

  The number of trees in the forest (maps to `size`).

- mtry:

  The number of variables to consider at each split (maps to `n_vars`).

- penalty:

  The regularization parameter (maps to `lambda`).

## Value

A parsnip model specification.

## See also

[`pprf`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/pprf.md)
for the underlying training function,
[`pp_tree`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/pp_tree.md)
for single trees

## Examples

``` r
if (FALSE) { # \dontrun{
library(parsnip)
spec <- pp_rand_forest(trees = 50, mtry = 2) %>% set_engine("ppforest2")
fit <- spec %>% fit(Species ~ ., data = iris)
predict(fit, iris)
predict(fit, iris, type = "prob")
} # }
```
