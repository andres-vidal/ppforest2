# Majority-vote leaf strategy.

Creates a leaf strategy that assigns the majority group label as the
leaf prediction. When groups are tied, the smallest label wins. This is
the default leaf strategy for classification.

## Usage

``` r
leaf_majority_vote()
```

## Value

A `leaf_strategy` object.

## See also

[`pptr`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/pptr.md),
[`pprf`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/pprf.md)

## Examples

``` r
leaf_majority_vote()
#> $name
#> [1] "majority_vote"
#> 
#> $display_name
#> [1] "Majority vote"
#> 
#> attr(,"class")
#> [1] "leaf_strategy"
```
