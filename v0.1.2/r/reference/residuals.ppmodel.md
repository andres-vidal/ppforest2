# Residuals from a regression ppforest2 model.

Returns `y - fitted(model)`. Only defined for regression models —
classification residuals have no canonical scalar form, so this method
errors on classification models rather than inventing a convention.

## Usage

``` r
# S3 method for class 'ppmodel'
residuals(object, ...)
```

## Arguments

- object:

  A `pprf` or `pptr` regression model.

- ...:

  Unused.

## Value

A numeric vector of length `nobs(object)`.

## See also

[`fitted.ppmodel`](https://andres-vidal.github.io/ppforest2/v0.1.2/r/reference/fitted.ppmodel.md)
