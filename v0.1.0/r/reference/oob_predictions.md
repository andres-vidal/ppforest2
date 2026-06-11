# Out-of-bag predictions for a random forest.

Returns predictions for each training row using only trees that did not
see that row in their bootstrap sample. Observations with no OOB tree
are represented as \`NA\` in both modes: \`NA\` at the factor level for
classification, and \`NA_real\_\` for regression. Filter with the
standard \`is.na()\` idiom.

## Usage

``` r
oob_predictions(model)
```

## Arguments

- model:

  A `pprf` forest model.

## Value

A factor (classification) or numeric vector (regression), length \`n\`.

## See also

[`oob_error`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/oob_error.md),
[`oob_samples`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/oob_samples.md)
