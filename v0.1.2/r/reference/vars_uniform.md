# Uniform random variable selection strategy.

Creates a variable selection strategy that randomly selects a subset of
variables at each split. Used with
[`pprf`](https://andres-vidal.github.io/ppforest2/v0.1.2/r/reference/pprf.md)
for random forests.

## Usage

``` r
vars_uniform(n_vars = NULL, p_vars = NULL)
```

## Arguments

- n_vars:

  The number of variables to consider at each split (integer). Cannot be
  used together with `p_vars`.

- p_vars:

  The proportion of variables to consider at each split (number between
  0 and 1, exclusive). Resolved to an integer count when the number of
  features is known. Cannot be used together with `n_vars`.

## Value

A `vars_strategy` object.

## Details

Exactly one of `n_vars` or `p_vars` may be specified. When `p_vars` is
used, it is stored as-is and resolved to an integer count later by
[`pprf`](https://andres-vidal.github.io/ppforest2/v0.1.2/r/reference/pprf.md)
once the number of features is known.

## See also

[`pprf`](https://andres-vidal.github.io/ppforest2/v0.1.2/r/reference/pprf.md),
[`vars_all`](https://andres-vidal.github.io/ppforest2/v0.1.2/r/reference/vars_all.md),
[`pp_pda`](https://andres-vidal.github.io/ppforest2/v0.1.2/r/reference/pp_pda.md),
[`cutpoint_mean_of_means`](https://andres-vidal.github.io/ppforest2/v0.1.2/r/reference/cutpoint_mean_of_means.md)

## Examples

``` r
# Select 2 variables at each split
vars_uniform(n_vars = 2)
#> $name
#> [1] "uniform"
#> 
#> $display_name
#> [1] "Uniform random"
#> 
#> $count
#> [1] 2
#> 
#> $p_vars
#> NULL
#> 
#> attr(,"class")
#> [1] "vars_strategy"

# Select half the variables at each split
vars_uniform(p_vars = 0.5)
#> $name
#> [1] "uniform"
#> 
#> $display_name
#> [1] "Uniform random"
#> 
#> $count
#> NULL
#> 
#> $p_vars
#> [1] 0.5
#> 
#> attr(,"class")
#> [1] "vars_strategy"

# Use with pprf
pprf(Species ~ ., data = iris, vars = vars_uniform(n_vars = 2))
#> 
#> Random Forest of Projection-Pursuit Oblique Decision Trees
#>   Call:        pprf(formula = Species ~ ., data = iris, vars = vars_uniform(n_vars = 2))
#>   Trees:       100
#>   Mode:        classification
#>   Group names: setosa, versicolor, virginica
#>   Formula:     Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width -     1
#> 
```
