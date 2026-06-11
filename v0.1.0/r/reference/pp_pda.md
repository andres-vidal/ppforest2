# PDA projection pursuit strategy.

Creates a Penalized Discriminant Analysis (PDA) projection pursuit
strategy for use with
[`pptr`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/pptr.md)
or
[`pprf`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/pprf.md).

## Usage

``` r
pp_pda(lambda = 0)
```

## Arguments

- lambda:

  A regularization parameter between 0 and 1. If `lambda = 0`, the model
  uses Linear Discriminant Analysis (LDA). If `lambda > 0`, the model
  uses Penalized Discriminant Analysis (PDA).

## Value

A `pp_strategy` object.

## See also

[`pptr`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/pptr.md),
[`pprf`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/pprf.md),
[`vars_uniform`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/vars_uniform.md),
[`vars_all`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/vars_all.md),
[`cutpoint_mean_of_means`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/cutpoint_mean_of_means.md)

## Examples

``` r
# PDA with lambda = 0.5
pp_pda(0.5)
#> $name
#> [1] "pda"
#> 
#> $display_name
#> [1] "PDA"
#> 
#> $lambda
#> [1] 0.5
#> 
#> attr(,"class")
#> [1] "pp_strategy"

# Use with pptr
pptr(Species ~ ., data = iris, pp = pp_pda(0.5))
#> 
#> Call: pptr(formula = Species ~ ., data = iris, pp = pp_pda(0.5))
#> 
#> Projection-Pursuit Oblique Decision Tree:
#> If ([ 0 -0.04 0.03 0.03 ] * x) < 0.01580044:
#>   Predict: setosa 
#> Else:
#>  If ([ 0 0.03 -0.06 -0.15 ] * x) < -0.4503323:
#>    Predict: virginica 
#>  Else:
#>    Predict: versicolor 
#> 

# Use with pprf
pprf(Species ~ ., data = iris, pp = pp_pda(0.5), vars = vars_uniform(n_vars = 2))
#> 
#> Random Forest of Projection-Pursuit Oblique Decision Trees
#>   Call:        pprf(formula = Species ~ ., data = iris, pp = pp_pda(0.5), vars = vars_uniform(n_vars = 2))
#>   Trees:       100
#>   Mode:        classification
#>   Group names: setosa, versicolor, virginica
#>   Formula:     Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width -     1
#> 
```
