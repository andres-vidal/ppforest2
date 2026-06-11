# Minimum-size stopping rule.

Creates a stopping rule that stops splitting when a node has fewer than
`min_size` observations. Used primarily for regression trees.

## Usage

``` r
stop_min_size(min_size = 5L)
```

## Arguments

- min_size:

  Minimum node size to allow a split (default: 5).

## Value

A `stop_strategy` object.

## See also

[`stop_min_variance`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/stop_min_variance.md),
[`stop_any`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/stop_any.md)

## Examples

``` r
stop_min_size(5)
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
```
