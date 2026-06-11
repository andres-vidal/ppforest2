# All-variables selection strategy.

Creates a variable selection strategy that uses all variables at each
split. This is the default for single trees
([`pptr`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/pptr.md)).

## Usage

``` r
vars_all()
```

## Value

A `vars_strategy` object.

## See also

[`pptr`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/pptr.md),
[`vars_uniform`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/vars_uniform.md),
[`pp_pda`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/pp_pda.md),
[`cutpoint_mean_of_means`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/cutpoint_mean_of_means.md)

## Examples

``` r
vars_all()
#> $name
#> [1] "all"
#> 
#> $display_name
#> [1] "All variables"
#> 
#> attr(,"class")
#> [1] "vars_strategy"
```
