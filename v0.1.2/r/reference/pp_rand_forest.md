# Parsnip model specification for pprf.

Creates a model specification for a Projection Pursuit random forest.
Use `set_engine("ppforest2")` to select the ppforest2 engine.

## Usage

``` r
pp_rand_forest(
  mode = "classification",
  trees = NULL,
  mtry = NULL,
  mtry_prop = NULL,
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

- mtry_prop:

  The proportion of variables to consider at each split (maps to
  `p_vars`). An alternative to `mtry` that expresses the feature
  subsample as a fraction in (0, 1\], tunable via the `dials`
  `mtry_prop()` parameter. Supply `mtry` or `mtry_prop`, not both.

- penalty:

  The regularization parameter (maps to `lambda`).

## Value

A parsnip model specification.

## See also

[`pprf`](https://andres-vidal.github.io/ppforest2/v0.1.2/r/reference/pprf.md)
for the underlying training function,
[`pp_tree`](https://andres-vidal.github.io/ppforest2/v0.1.2/r/reference/pp_tree.md)
for single trees

## Examples

``` r
# \donttest{
if (requireNamespace("parsnip", quietly = TRUE)) {
  library(parsnip)
  spec <- pp_rand_forest(trees = 50, mtry = 2) %>% set_engine("ppforest2")
  fit <- spec %>% fit(Species ~ ., data = iris)
  predict(fit, iris)
  predict(fit, iris, type = "prob")
}
#> # A tibble: 150 × 3
#>    .pred_setosa .pred_versicolor .pred_virginica
#>           <dbl>            <dbl>           <dbl>
#>  1            1                0               0
#>  2            1                0               0
#>  3            1                0               0
#>  4            1                0               0
#>  5            1                0               0
#>  6            1                0               0
#>  7            1                0               0
#>  8            1                0               0
#>  9            1                0               0
#> 10            1                0               0
#> # ℹ 140 more rows
# }
```
