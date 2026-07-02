# Predicts numeric responses from a pptr model (regression mode).

Predicts numeric responses from a pptr model (regression mode).

## Usage

``` r
# S3 method for class 'pptr_regression'
predict(object, new_data = NULL, type = NULL, ...)
```

## Arguments

- object:

  A `pptr_regression` model.

- new_data:

  A data frame or matrix of new observations.

- type:

  Must be `"response"` (default).

- ...:

  Backward-compat positional \`new_data\`.

## Value

A numeric vector.

## See also

[`pptr`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/pptr.md),
[`predict.pptr_classification`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/predict.pptr_classification.md)
