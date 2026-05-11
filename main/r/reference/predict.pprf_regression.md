# Predicts numeric responses from a pprf model (regression mode).

Predicts numeric responses from a pprf model (regression mode).

## Usage

``` r
# S3 method for class 'pprf_regression'
predict(object, new_data = NULL, type = NULL, ...)
```

## Arguments

- object:

  A `pprf_regression` model.

- new_data:

  A data frame or matrix of new observations.

- type:

  Must be `"response"` (default).

- ...:

  For backward compatibility, the first positional argument is treated
  as `new_data` when `new_data` is `NULL`.

## Value

A numeric vector of mean predictions across the forest's trees.

## See also

[`pprf`](https://andres-vidal.github.io/ppforest2/main/r/reference/pprf.md),
[`predict.pprf_classification`](https://andres-vidal.github.io/ppforest2/main/r/reference/predict.pprf_classification.md)
