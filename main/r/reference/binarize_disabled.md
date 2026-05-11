# Disabled binarization strategy (placeholder).

Placeholder binarizer for specs where binarization never fires — notably
regression, where \`grouping_by_cutpoint()\` always produces a 2-group
partition at each node. Selecting `binarize_disabled()` documents that
intent explicitly; if binarization is ever invoked at runtime with this
strategy configured, training aborts with a clear error rather than
silently passing through. Default for regression specs.

## Usage

``` r
binarize_disabled()
```

## Value

A `binarize_strategy` object.

## See also

[`binarize_largest_gap`](https://andres-vidal.github.io/ppforest2/main/r/reference/binarize_largest_gap.md)

## Examples

``` r
binarize_disabled()
#> $name
#> [1] "disabled"
#> 
#> $display_name
#> [1] "Disabled"
#> 
#> attr(,"class")
#> [1] "binarize_strategy"
```
