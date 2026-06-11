# Pure-node stopping rule.

Creates a stopping rule that stops splitting when a node contains only
one group. This is the default stopping rule for classification.

## Usage

``` r
stop_pure_node()
```

## Value

A `stop_strategy` object.

## See also

[`pptr`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/pptr.md),
[`pprf`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/pprf.md)

## Examples

``` r
stop_pure_node()
#> $name
#> [1] "pure_node"
#> 
#> $display_name
#> [1] "Pure node"
#> 
#> attr(,"class")
#> [1] "stop_strategy"
```
