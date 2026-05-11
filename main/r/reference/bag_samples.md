# In-bag row indices per tree.

Returns a list where element \`i\` is the integer vector of row indices
(1-based, with replacement) drawn into the bootstrap sample of tree
\`i\`.

## Usage

``` r
bag_samples(model)
```

## Arguments

- model:

  A `pprf` forest model.

## Value

A list of integer vectors, one per tree.

## See also

[`oob_samples`](https://andres-vidal.github.io/ppforest2/main/r/reference/oob_samples.md)
