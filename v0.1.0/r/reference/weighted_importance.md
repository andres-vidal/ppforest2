# Weighted projection variable importance for a random forest.

Weights each tree's projection-based importance by a per-tree OOB
quality score — \`1 - error_rate\` in \`\[0, 1\]\` for classification,
and \`max(0, 1 - NMSE)\` in \`\[0, 1\]\` for regression — then
aggregates \`I_s × \|a_j\|\` over splits. Computed lazily from the
training data stored on the model; the result is cached.

## Usage

``` r
weighted_importance(model)
```

## Arguments

- model:

  A `pprf` forest model.

## Value

A non-negative numeric vector, one entry per feature.

## Details

\*\*Sign semantics.\*\* Entries are non-negative by construction
(weights and per-split contributions are both non-negative). A zero
entry means "this feature never appeared in a weighted OOB-contributing
split," not "within noise." Contrast with
[`permuted_importance`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/permuted_importance.md),
where negative values are meaningful. Do not re-normalize — rely on the
ranking.
