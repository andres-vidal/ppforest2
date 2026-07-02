# Label-based grouping strategy.

Creates a grouping strategy that routes all observations of a group to
the same child node. This is the default grouping strategy for
classification.

## Usage

``` r
grouping_by_label()
```

## Value

A `grouping_strategy` object.

## See also

[`pptr`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/pptr.md),
[`pprf`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/pprf.md)

## Examples

``` r
grouping_by_label()
#> $name
#> [1] "by_label"
#> 
#> $display_name
#> [1] "By label"
#> 
#> attr(,"class")
#> [1] "grouping_strategy"
```
