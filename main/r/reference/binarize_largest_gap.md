# Largest-gap binarization strategy.

Creates a binarization strategy that reduces multiclass nodes to binary
by projecting group means and splitting at the largest gap. Default for
classification specs.

## Usage

``` r
binarize_largest_gap()
```

## Value

A `binarize_strategy` object.

## See also

[`pptr`](https://andres-vidal.github.io/ppforest2/main/r/reference/pptr.md),
[`pprf`](https://andres-vidal.github.io/ppforest2/main/r/reference/pprf.md),
[`binarize_disabled`](https://andres-vidal.github.io/ppforest2/main/r/reference/binarize_disabled.md)

## Examples

``` r
binarize_largest_gap()
#> $name
#> [1] "largest_gap"
#> 
#> $display_name
#> [1] "Largest gap"
#> 
#> attr(,"class")
#> [1] "binarize_strategy"
```
