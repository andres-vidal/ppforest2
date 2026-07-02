# Predicts labels or per-group one-hot proportions from a pptr model (classification mode).

Predicts labels or per-group one-hot proportions from a pptr model
(classification mode).

## Usage

``` r
# S3 method for class 'pptr_classification'
predict(object, new_data = NULL, type = NULL, ...)
```

## Arguments

- object:

  A `pptr_classification` model.

- new_data:

  A data frame or matrix of new observations. If `NULL`, the first
  positional argument in `...` is used for backward compatibility.

- type:

  `"class"` (default) returns a factor of predicted labels; `"prob"`
  returns a data frame with 1.0 for the predicted group and 0.0
  elsewhere.

- ...:

  Backward-compat positional \`new_data\`.

## Value

A factor or data frame.

## See also

[`pptr`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/pptr.md),
[`predict.pptr_regression`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/predict.pptr_regression.md)
