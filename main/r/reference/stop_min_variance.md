# Minimum-variance stopping rule.

Creates a stopping rule that stops splitting when the within-node
response variance falls below `threshold`. Used primarily for regression
trees; requires a continuous response.

## Usage

``` r
stop_min_variance(threshold = 0.01)
```

## Arguments

- threshold:

  Variance threshold below which to stop splitting (default: 0.01).

## Value

A `stop_strategy` object.

## See also

[`stop_min_size`](https://andres-vidal.github.io/ppforest2/main/r/reference/stop_min_size.md),
[`stop_any`](https://andres-vidal.github.io/ppforest2/main/r/reference/stop_any.md)

## Examples

``` r
stop_min_variance(0.01)
#> $name
#> [1] "min_variance"
#> 
#> $threshold
#> [1] 0.01
#> 
#> $display_name
#> [1] "Min variance (   0.01)"
#> 
#> attr(,"class")
#> [1] "stop_strategy"
```
