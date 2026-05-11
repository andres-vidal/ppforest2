# Out-of-bag error for a random forest.

Computes (or returns the cached) OOB error using the training data
stored on the model. For classification, this is the misclassification
rate in \`\[0, 1\]\`. For regression, it is the mean squared error
against the continuous response.

## Usage

``` r
oob_error(model)
```

## Arguments

- model:

  A `pprf` forest model.

## Value

A numeric scalar in \`\[0, 1\]\` for classification or \`\[0, Inf)\` for
regression. Returns \`NA_real\_\` when no observation has any out-of-bag
tree (e.g. a degenerate forest where every tree saw every row). Callers
should check with \`is.na()\` rather than comparing against a sentinel
value; in earlier versions this condition was signalled as \`-1\`, which
was not distinguishable from a (mathematically impossible but
representable) error rate.

## See also

[`oob_predictions`](https://andres-vidal.github.io/ppforest2/main/r/reference/oob_predictions.md),
[`oob_samples`](https://andres-vidal.github.io/ppforest2/main/r/reference/oob_samples.md)
