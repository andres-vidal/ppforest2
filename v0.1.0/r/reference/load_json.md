# Load a model from a JSON file.

Deserializes a `pptr` or `pprf` model from a JSON file. The model can be
used for prediction immediately. If variable importance metrics and OOB
fields were saved, they are restored into the model's lazy-computation
cache so later accessor calls
([`oob_error()`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/oob_error.md),
[`permuted_importance()`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/permuted_importance.md),
etc.) return the stored values without recomputation.

## Usage

``` r
load_json(path)
```

## Arguments

- path:

  File path to read the JSON from.

## Value

A `pptr` or `pprf` model (with the appropriate `_classification` /
`_regression` subclass).

## Details

Note that `formula`, `x`, and `y` are not stored in the JSON and will be
`NULL` on the loaded model. Formula-based prediction and any accessor
that requires training data will error unless the user re-attaches those
fields.

## See also

[`save_json`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/save_json.md),
[`pptr`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/pptr.md),
[`pprf`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/pprf.md)

## Examples

``` r
model <- pptr(Species ~ ., data = iris, seed = 0)
path <- tempfile(fileext = ".json")
save_json(model, path)
loaded <- load_json(path)
predict(loaded, as.matrix(iris[, 1:4]))
#>   [1] setosa     setosa     setosa     setosa     setosa     setosa    
#>   [7] setosa     setosa     setosa     setosa     setosa     setosa    
#>  [13] setosa     setosa     setosa     setosa     setosa     setosa    
#>  [19] setosa     setosa     setosa     setosa     setosa     setosa    
#>  [25] setosa     setosa     setosa     setosa     setosa     setosa    
#>  [31] setosa     setosa     setosa     setosa     setosa     setosa    
#>  [37] setosa     setosa     setosa     setosa     setosa     setosa    
#>  [43] setosa     setosa     setosa     setosa     setosa     setosa    
#>  [49] setosa     setosa     versicolor versicolor versicolor versicolor
#>  [55] versicolor versicolor versicolor versicolor versicolor versicolor
#>  [61] versicolor versicolor versicolor versicolor versicolor versicolor
#>  [67] versicolor versicolor versicolor versicolor virginica  versicolor
#>  [73] versicolor versicolor versicolor versicolor versicolor versicolor
#>  [79] versicolor versicolor versicolor versicolor versicolor virginica 
#>  [85] versicolor versicolor versicolor versicolor versicolor versicolor
#>  [91] versicolor versicolor versicolor versicolor versicolor versicolor
#>  [97] versicolor versicolor versicolor versicolor virginica  virginica 
#> [103] virginica  virginica  virginica  virginica  virginica  virginica 
#> [109] virginica  virginica  virginica  virginica  virginica  virginica 
#> [115] virginica  virginica  virginica  virginica  virginica  virginica 
#> [121] virginica  virginica  virginica  virginica  virginica  virginica 
#> [127] virginica  virginica  virginica  virginica  virginica  virginica 
#> [133] virginica  versicolor virginica  virginica  virginica  virginica 
#> [139] virginica  virginica  virginica  virginica  virginica  virginica 
#> [145] virginica  virginica  virginica  virginica  virginica  virginica 
#> Levels: setosa versicolor virginica
```
