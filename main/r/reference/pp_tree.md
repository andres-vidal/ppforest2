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

[`pptr`](https://andres-vidal.github.io/ppforest2/main/r/reference/pptr.md)
for the underlying training function,
[`pp_rand_forest`](https://andres-vidal.github.io/ppforest2/main/r/reference/pp_rand_forest.md)
for forests

## Examples

``` r
if (FALSE) { # \dontrun{
library(parsnip)
spec <- pp_tree(penalty = 0) %>% set_engine("ppforest2")
fit <- spec %>% fit(Species ~ ., data = iris)
predict(fit, iris)
} # }
```
