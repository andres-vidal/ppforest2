# Cutpoint-based grouping strategy (regression).

Creates a grouping strategy for regression trees: observations are split
by the cutpoint in projected space, then each child's observations are
sorted by the continuous response and median-split into 2 new groups for
the next projection-pursuit step. This is the default grouping strategy
for regression.

## Usage

``` r
grouping_by_cutpoint()
```

## Value

A `grouping_strategy` object.

## See also

[`pptr`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/pptr.md),
[`pprf`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/pprf.md),
[`grouping_by_label`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/grouping_by_label.md)

## Examples

``` r
grouping_by_cutpoint()
#> $name
#> [1] "by_cutpoint"
#> 
#> $display_name
#> [1] "By cutpoint"
#> 
#> attr(,"class")
#> [1] "grouping_strategy"
```
