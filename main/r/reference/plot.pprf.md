# Plot a pprf model.

Visualizes a pprf model. By default, shows variable importance with one
plot per metric side by side. Use `metric` to show a single importance
metric.

## Usage

``` r
# S3 method for class 'pprf'
plot(x, type = "importance", metric = NULL, tree_index = 1L, node = 1L, ...)
```

## Arguments

- x:

  A pprf model.

- type:

  Character string specifying the plot type. `"importance"` (default)
  shows variable importance, `"structure"` shows a specific tree with
  embedded histograms, `"projection"` shows projected data at a node,
  `"boundaries"` shows decision boundaries of a specific tree.

- metric:

  Character string selecting a single importance metric to plot:
  `"projections"`, `"weighted"`, or `"permuted"`. `NULL` (default) shows
  all available metrics side by side in separate panels. Only used when
  `type = "importance"`.

- tree_index:

  Integer index of the tree to plot (1-based). Only used when
  `type = "structure"`, `type = "projection"`, or `type = "boundaries"`.
  Defaults to 1.

- node:

  Integer index of the node for projection plots. Defaults to 1 (root).
  Only used when `type = "projection"`.

- ...:

  Additional arguments passed to the internal plotting function.

## Value

A ggplot2-compatible object (invisibly). The importance grid returns a
patchwork object that works with
[`ggplot2::ggsave()`](https://ggplot2.tidyverse.org/reference/ggsave.html).

## Examples

``` r
# \donttest{
if (requireNamespace("ggplot2", quietly = TRUE) &&
    requireNamespace("patchwork", quietly = TRUE)) {
  forest <- pprf(Species ~ ., data = iris, size = 10)
  plot(forest)                                    # all metrics side by side
  plot(forest, metric = "permuted")               # single metric
  plot(forest, type = "structure", tree_index = 1)
  plot(forest, type = "projection", tree_index = 1)
}




# }
```
