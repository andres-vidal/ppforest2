# Number of observations used to fit a ppforest2 model.

Implements the standard
[`stats::nobs()`](https://rdrr.io/r/stats/nobs.html) contract so
downstream tools ([`step()`](https://rdrr.io/r/stats/step.html), broom's
[`glance()`](https://generics.r-lib.org/reference/glance.html),
information criteria) can ask for the training-sample size.

## Usage

``` r
# S3 method for class 'ppmodel'
nobs(object, ...)
```

## Arguments

- object:

  A `pptr` or `pprf` model.

- ...:

  Unused.

## Value

Integer scalar. Returns `NA_integer_` for models loaded from JSON
without their original training data.
