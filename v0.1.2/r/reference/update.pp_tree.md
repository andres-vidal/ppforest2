# Update a `pp_tree` model specification.

Companion to
[`update.pp_rand_forest`](https://andres-vidal.github.io/ppforest2/v0.1.2/r/reference/update.pp_rand_forest.md)
— exists for the same reason
([`tune::tune_grid()`](https://tune.tidymodels.org/reference/tune_grid.html)
relies on [`update()`](https://rdrr.io/r/stats/update.html) to finalise
the spec at each grid point).

## Usage

``` r
# S3 method for class 'pp_tree'
update(object, parameters = NULL, penalty = NULL, fresh = FALSE, ...)
```

## Arguments

- object:

  A `pp_tree` model specification.

- parameters:

  A named list of parameters to update (alternative to passing them as
  args).

- penalty:

  A new value for the regularisation parameter.

- fresh:

  If `TRUE`, drop any previously-set arg whose new value is not
  provided.

- ...:

  Engine-specific arguments to update.

## Value

An updated `pp_tree` model specification.
