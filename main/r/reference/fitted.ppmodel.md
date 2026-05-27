# Fitted (in-sample) predictions from a ppforest2 model.

Returns predictions on the training data — a factor for classification,
a numeric vector for regression. For forests, training predictions are
optimistic; use
[`oob_predictions()`](https://andres-vidal.github.io/ppforest2/main/r/reference/oob_predictions.md)
for an honest estimate.

## Usage

``` r
# S3 method for class 'ppmodel'
fitted(object, ...)
```

## Arguments

- object:

  A `pptr` or `pprf` model.

- ...:

  Unused.

## Value

A factor (classification) or numeric vector (regression), length equal
to the number of training observations.

## See also

[`residuals.ppmodel`](https://andres-vidal.github.io/ppforest2/main/r/reference/residuals.ppmodel.md),
[`oob_predictions`](https://andres-vidal.github.io/ppforest2/main/r/reference/oob_predictions.md)
