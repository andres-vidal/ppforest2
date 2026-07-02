# Projection-coefficient variable importance.

The projection-based importance (VI2): each split's scaled absolute
projection coefficients (\`\|a_j\| \* sigma_j\`) aggregated into a
per-feature score, averaged over the non-degenerate trees of a forest.
Unlike
[`permuted_importance`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/permuted_importance.md)
and
[`weighted_importance`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/weighted_importance.md),
this measure is not OOB-based — it depends only on the fitted projector
geometry, is computed eagerly at fit time (cheap), and is available for
both single trees (`pptr`) and forests (`pprf`).

## Usage

``` r
projection_importance(model)
```

## Arguments

- model:

  A `pptr` or `pprf` model.

## Value

A non-negative numeric vector, one entry per feature.

## Details

\*\*Sign semantics.\*\* Entries are non-negative by construction
(absolute coefficients scaled by each feature's standard deviation).
Rely on the ranking rather than re-normalizing.

## See also

[`permuted_importance`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/permuted_importance.md),
[`weighted_importance`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/weighted_importance.md)
