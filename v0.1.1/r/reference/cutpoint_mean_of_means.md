# Mean-of-means split cutpoint strategy.

Creates a split cutpoint strategy that splits at the midpoint between
group means. This is the default (and currently only) split cutpoint.

## Usage

``` r
cutpoint_mean_of_means()
```

## Value

A `cutpoint_strategy` object.

## See also

[`pptr`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/pptr.md),
[`pprf`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/pprf.md),
[`pp_pda`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/pp_pda.md),
[`vars_uniform`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/vars_uniform.md)

## Examples

``` r
cutpoint_mean_of_means()
#> $name
#> [1] "mean_of_means"
#> 
#> $display_name
#> [1] "Mean of means"
#> 
#> attr(,"class")
#> [1] "cutpoint_strategy"
```
