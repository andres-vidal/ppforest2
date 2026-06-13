# Predicts labels or vote proportions from a pprf model (classification mode).

Predicts labels or vote proportions from a pprf model (classification
mode).

## Usage

``` r
# S3 method for class 'pprf_classification'
predict(object, new_data = NULL, type = NULL, ...)
```

## Arguments

- object:

  A `pprf_classification` model.

- new_data:

  A data frame or matrix of new observations. If `NULL`, the first
  positional argument in `...` is used for backward compatibility.

- type:

  The type of prediction: `"class"` (default) returns a factor of
  predicted labels, `"prob"` returns a data frame of vote proportions.

- ...:

  For backward compatibility, the first positional argument is treated
  as `new_data` when `new_data` is `NULL`.

## Value

If `type = "class"`, a factor of predicted labels. If `type = "prob"`, a
data frame with one column per group, each row summing to 1.

## See also

[`pprf`](https://andres-vidal.github.io/ppforest2/main/r/reference/pprf.md),
[`predict.pprf_regression`](https://andres-vidal.github.io/ppforest2/main/r/reference/predict.pprf_regression.md)

## Examples

``` r
model <- pprf(Species ~ ., data = iris)
predict(model, iris)
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
#>  [73] versicolor versicolor versicolor versicolor versicolor virginica 
#>  [79] versicolor versicolor versicolor versicolor versicolor versicolor
#>  [85] versicolor versicolor versicolor versicolor versicolor versicolor
#>  [91] versicolor versicolor versicolor versicolor versicolor versicolor
#>  [97] versicolor versicolor versicolor versicolor virginica  virginica 
#> [103] virginica  virginica  virginica  virginica  versicolor virginica 
#> [109] virginica  virginica  virginica  virginica  virginica  virginica 
#> [115] virginica  virginica  virginica  virginica  virginica  versicolor
#> [121] virginica  virginica  virginica  virginica  virginica  virginica 
#> [127] virginica  virginica  virginica  virginica  virginica  virginica 
#> [133] virginica  versicolor versicolor virginica  virginica  virginica 
#> [139] virginica  virginica  virginica  virginica  virginica  virginica 
#> [145] virginica  virginica  virginica  virginica  virginica  virginica 
#> Levels: setosa versicolor virginica
predict(model, iris, type = "prob")
#>     setosa versicolor virginica
#> 1     1.00       0.00      0.00
#> 2     1.00       0.00      0.00
#> 3     1.00       0.00      0.00
#> 4     1.00       0.00      0.00
#> 5     1.00       0.00      0.00
#> 6     1.00       0.00      0.00
#> 7     1.00       0.00      0.00
#> 8     1.00       0.00      0.00
#> 9     1.00       0.00      0.00
#> 10    1.00       0.00      0.00
#> 11    1.00       0.00      0.00
#> 12    1.00       0.00      0.00
#> 13    1.00       0.00      0.00
#> 14    1.00       0.00      0.00
#> 15    1.00       0.00      0.00
#> 16    1.00       0.00      0.00
#> 17    1.00       0.00      0.00
#> 18    1.00       0.00      0.00
#> 19    1.00       0.00      0.00
#> 20    1.00       0.00      0.00
#> 21    1.00       0.00      0.00
#> 22    1.00       0.00      0.00
#> 23    1.00       0.00      0.00
#> 24    1.00       0.00      0.00
#> 25    1.00       0.00      0.00
#> 26    1.00       0.00      0.00
#> 27    1.00       0.00      0.00
#> 28    1.00       0.00      0.00
#> 29    1.00       0.00      0.00
#> 30    1.00       0.00      0.00
#> 31    1.00       0.00      0.00
#> 32    1.00       0.00      0.00
#> 33    1.00       0.00      0.00
#> 34    1.00       0.00      0.00
#> 35    1.00       0.00      0.00
#> 36    1.00       0.00      0.00
#> 37    1.00       0.00      0.00
#> 38    1.00       0.00      0.00
#> 39    1.00       0.00      0.00
#> 40    1.00       0.00      0.00
#> 41    1.00       0.00      0.00
#> 42    0.85       0.15      0.00
#> 43    1.00       0.00      0.00
#> 44    1.00       0.00      0.00
#> 45    1.00       0.00      0.00
#> 46    1.00       0.00      0.00
#> 47    1.00       0.00      0.00
#> 48    1.00       0.00      0.00
#> 49    1.00       0.00      0.00
#> 50    1.00       0.00      0.00
#> 51    0.00       0.80      0.20
#> 52    0.00       0.81      0.19
#> 53    0.00       0.65      0.35
#> 54    0.00       1.00      0.00
#> 55    0.00       0.81      0.19
#> 56    0.00       1.00      0.00
#> 57    0.00       0.80      0.20
#> 58    0.14       0.86      0.00
#> 59    0.00       0.81      0.19
#> 60    0.00       1.00      0.00
#> 61    0.14       0.86      0.00
#> 62    0.00       1.00      0.00
#> 63    0.00       1.00      0.00
#> 64    0.00       0.99      0.01
#> 65    0.00       1.00      0.00
#> 66    0.00       0.81      0.19
#> 67    0.00       1.00      0.00
#> 68    0.00       1.00      0.00
#> 69    0.00       1.00      0.00
#> 70    0.00       1.00      0.00
#> 71    0.00       0.41      0.59
#> 72    0.00       1.00      0.00
#> 73    0.00       0.85      0.15
#> 74    0.00       1.00      0.00
#> 75    0.00       0.81      0.19
#> 76    0.00       0.81      0.19
#> 77    0.00       0.79      0.21
#> 78    0.00       0.05      0.95
#> 79    0.00       1.00      0.00
#> 80    0.00       1.00      0.00
#> 81    0.00       1.00      0.00
#> 82    0.03       0.97      0.00
#> 83    0.00       1.00      0.00
#> 84    0.00       0.65      0.35
#> 85    0.13       0.87      0.00
#> 86    0.10       0.76      0.14
#> 87    0.00       0.80      0.20
#> 88    0.00       0.99      0.01
#> 89    0.00       1.00      0.00
#> 90    0.00       1.00      0.00
#> 91    0.00       1.00      0.00
#> 92    0.00       0.95      0.05
#> 93    0.00       1.00      0.00
#> 94    0.14       0.86      0.00
#> 95    0.00       1.00      0.00
#> 96    0.00       1.00      0.00
#> 97    0.00       1.00      0.00
#> 98    0.00       0.94      0.06
#> 99    0.26       0.74      0.00
#> 100   0.00       1.00      0.00
#> 101   0.00       0.00      1.00
#> 102   0.00       0.19      0.81
#> 103   0.00       0.00      1.00
#> 104   0.00       0.04      0.96
#> 105   0.00       0.00      1.00
#> 106   0.00       0.00      1.00
#> 107   0.01       0.75      0.24
#> 108   0.00       0.00      1.00
#> 109   0.00       0.01      0.99
#> 110   0.00       0.00      1.00
#> 111   0.00       0.00      1.00
#> 112   0.00       0.09      0.91
#> 113   0.00       0.00      1.00
#> 114   0.00       0.23      0.77
#> 115   0.00       0.19      0.81
#> 116   0.00       0.00      1.00
#> 117   0.00       0.00      1.00
#> 118   0.00       0.00      1.00
#> 119   0.00       0.00      1.00
#> 120   0.00       0.79      0.21
#> 121   0.00       0.00      1.00
#> 122   0.00       0.31      0.69
#> 123   0.00       0.00      1.00
#> 124   0.00       0.27      0.73
#> 125   0.00       0.00      1.00
#> 126   0.00       0.00      1.00
#> 127   0.00       0.43      0.57
#> 128   0.00       0.27      0.73
#> 129   0.00       0.03      0.97
#> 130   0.00       0.26      0.74
#> 131   0.00       0.00      1.00
#> 132   0.00       0.00      1.00
#> 133   0.00       0.03      0.97
#> 134   0.00       0.67      0.33
#> 135   0.00       0.71      0.29
#> 136   0.00       0.00      1.00
#> 137   0.00       0.00      1.00
#> 138   0.00       0.00      1.00
#> 139   0.00       0.43      0.57
#> 140   0.00       0.00      1.00
#> 141   0.00       0.00      1.00
#> 142   0.00       0.00      1.00
#> 143   0.00       0.19      0.81
#> 144   0.00       0.00      1.00
#> 145   0.00       0.00      1.00
#> 146   0.00       0.00      1.00
#> 147   0.00       0.20      0.80
#> 148   0.00       0.00      1.00
#> 149   0.00       0.01      0.99
#> 150   0.00       0.19      0.81
```
