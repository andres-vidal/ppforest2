# Maximum-depth stopping rule.

Creates a stopping rule that stops splitting when a node's depth reaches
`max_depth`. Depth is zero-based at the root, so `max_depth(k)` allows
at most `k + 1` levels. Mode-agnostic: useful for bounding tree
complexity in both classification and regression trees.

## Usage

``` r
stop_max_depth(max_depth)
```

## Arguments

- max_depth:

  Maximum depth (non-negative integer; 0 produces a root-only stump).

## Value

A `stop_strategy` object.

## See also

[`stop_min_size`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/stop_min_size.md),
[`stop_any`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/stop_any.md)

## Examples

``` r
stop_max_depth(5)
#> $name
#> [1] "max_depth"
#> 
#> $max_depth
#> [1] 5
#> 
#> $display_name
#> [1] "Max depth (5)"
#> 
#> attr(,"class")
#> [1] "stop_strategy"
```
