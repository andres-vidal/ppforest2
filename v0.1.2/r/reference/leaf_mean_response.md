# Mean-response leaf strategy.

Creates a leaf strategy that predicts the mean of the continuous
response values for the observations in the leaf. Used for regression
trees; requires a numeric response.

## Usage

``` r
leaf_mean_response()
```

## Value

A `leaf_strategy` object.

## See also

[`pptr`](https://andres-vidal.github.io/ppforest2/v0.1.2/r/reference/pptr.md),
[`pprf`](https://andres-vidal.github.io/ppforest2/v0.1.2/r/reference/pprf.md),
[`leaf_majority_vote`](https://andres-vidal.github.io/ppforest2/v0.1.2/r/reference/leaf_majority_vote.md)

## Examples

``` r
leaf_mean_response()
#> $name
#> [1] "mean_response"
#> 
#> $display_name
#> [1] "Mean response"
#> 
#> attr(,"class")
#> [1] "leaf_strategy"
```
