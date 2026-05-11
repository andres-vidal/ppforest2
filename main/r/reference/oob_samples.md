# Out-of-bag row indices per tree.

Returns a list where element \`i\` is the integer vector of row indices
(1-based) that were \*\*not\*\* in the bootstrap sample of tree \`i\`.

## Usage

``` r
oob_samples(model)
```

## Arguments

- model:

  A `pprf` forest model.

## Value

A list of integer vectors, one per tree.

## See also

[`bag_samples`](https://andres-vidal.github.io/ppforest2/main/r/reference/bag_samples.md),
[`oob_error`](https://andres-vidal.github.io/ppforest2/main/r/reference/oob_error.md)
