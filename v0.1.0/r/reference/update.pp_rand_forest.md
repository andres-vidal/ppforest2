# Update a `pp_rand_forest` model specification.

Implements parsnip's `update` protocol so that
[`tune::tune_grid()`](https://tune.tidymodels.org/reference/tune_grid.html)
(and any other caller that finalises a spec via
[`update()`](https://rdrr.io/r/stats/update.html)) can fill in the tuned
values for `trees`, `mtry`, and `penalty`. Without this method,
[`update()`](https://rdrr.io/r/stats/update.html) falls back to
[`stats::update.default`](https://rdrr.io/r/stats/update.html) and fails
with "need an object with call component".

## Usage

``` r
# S3 method for class 'pp_rand_forest'
update(
  object,
  parameters = NULL,
  trees = NULL,
  mtry = NULL,
  mtry_prop = NULL,
  penalty = NULL,
  fresh = FALSE,
  ...
)
```

## Arguments

- object:

  A `pp_rand_forest` model specification.

- parameters:

  A named list of parameters to update (alternative to passing them as
  args).

- trees, mtry, mtry_prop, penalty:

  New values for the corresponding parameters.

- fresh:

  If `TRUE`, drop any previously-set arg whose new value is not
  provided.

- ...:

  Engine-specific arguments to update.

## Value

An updated `pp_rand_forest` model specification.
