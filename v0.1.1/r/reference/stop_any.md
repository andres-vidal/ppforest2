# Composite stopping rule (logical OR).

Creates a composite stopping rule that fires when any of the child rules
fires. Useful for combining multiple criteria, e.g.
`stop_any(stop_min_size(5), stop_min_variance(0.01))` for regression.

## Usage

``` r
stop_any(...)
```

## Arguments

- ...:

  Two or more `stop_strategy` objects to combine.

## Value

A `stop_strategy` object.

## See also

[`stop_min_size`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/stop_min_size.md),
[`stop_min_variance`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/stop_min_variance.md),
[`stop_pure_node`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/stop_pure_node.md)

## Examples

``` r
stop_any(stop_min_size(5), stop_min_variance(0.01))
#> $name
#> [1] "any"
#> 
#> $rules
#> $rules[[1]]
#> $name
#> [1] "min_size"
#> 
#> $min_size
#> [1] 5
#> 
#> $display_name
#> [1] "Min size (5)"
#> 
#> attr(,"class")
#> [1] "stop_strategy"
#> 
#> $rules[[2]]
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
#> 
#> 
#> $display_name
#> [1] "Any(Min size (5), Min variance (   0.01))"
#> 
#> attr(,"class")
#> [1] "stop_strategy"
```
