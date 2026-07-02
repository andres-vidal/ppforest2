# Parsnip model specification for pptr.

Creates a model specification for a single Projection Pursuit decision
tree. Use `set_engine("ppforest2")` to select the ppforest2 engine.

## Usage

``` r
pp_tree(mode = "classification", penalty = NULL)
```

## Arguments

- mode:

  A character string for the model type. Either `"classification"` or
  `"regression"`.

- penalty:

  The regularization parameter (maps to `lambda`).

## Value

A parsnip model specification.

## See also

[`pptr`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/pptr.md)
for the underlying training function,
[`pp_rand_forest`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/pp_rand_forest.md)
for forests

## Examples

``` r
# \donttest{
if (requireNamespace("parsnip", quietly = TRUE)) {
  library(parsnip)
  spec <- pp_tree(penalty = 0) %>% set_engine("ppforest2")
  fit <- spec %>% fit(Species ~ ., data = iris)
  predict(fit, iris)
}
#> # A tibble: 150 × 1
#>    .pred_class
#>    <fct>      
#>  1 setosa     
#>  2 setosa     
#>  3 setosa     
#>  4 setosa     
#>  5 setosa     
#>  6 setosa     
#>  7 setosa     
#>  8 setosa     
#>  9 setosa     
#> 10 setosa     
#> # ℹ 140 more rows
# }
```
