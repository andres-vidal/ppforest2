# Trains a Project-Pursuit oblique decision tree.

This function trains a Project-Pursuit oblique decision tree using
either a formula and data frame interface or a matrix-based interface.
When using the formula interface, specify the model formula and the data
frame containing the variables. For the matrix-based interface, provide
matrices for the features and labels directly. If `lambda = 0`, the
model is trained using Linear Discriminant Analysis (LDA). If
`lambda > 0`, the model is trained using Penalized Discriminant Analysis
(PDA).

## Usage

``` r
pptr(
  formula = NULL,
  data = NULL,
  x = NULL,
  y = NULL,
  mode = NULL,
  lambda = 0,
  seed = NULL,
  pp = NULL,
  cutpoint = NULL,
  stop = NULL,
  binarize = NULL,
  grouping = NULL,
  leaf = NULL
)
```

## Arguments

- formula:

  A formula of the form `y ~ x1 + x2 + ...`, where `y` is a vector of
  labels and `x1`, `x2`, ... are the features.

- data:

  A data frame containing the variables in the formula.

- x:

  A matrix containing the features for each observation.

- y:

  A matrix containing the labels for each observation.

- mode:

  Training mode: either `"classification"` or `"regression"`. When
  `NULL` (default), mode is auto-detected from `y`'s type — factor or
  character vectors trigger classification, numeric vectors trigger
  regression. Setting it explicitly is useful for the
  binary-integer-labels case (`mode = "classification"` with integer 0/1
  labels) and for failing fast on a type mismatch (`mode = "regression"`
  with a factor `y` errors immediately).

- lambda:

  A regularization parameter. If `lambda = 0`, the model is trained
  using Linear Discriminant Analysis (LDA). If `lambda > 0`, the model
  is trained using Penalized Discriminant Analysis (PDA). Cannot be used
  together with `pp`.

- seed:

  An optional integer seed for reproducibility. If `NULL` (default), a
  seed is drawn from R's RNG, so
  [`set.seed()`](https://rdrr.io/r/base/Random.html) controls
  reproducibility. If an integer is provided, that value is used
  directly.

- pp:

  A projection pursuit strategy object created by
  [`pp_pda`](https://andres-vidal.github.io/ppforest2/main/r/reference/pp_pda.md).
  Cannot be used together with `lambda`.

- cutpoint:

  A split cutpoint strategy object created by
  [`cutpoint_mean_of_means`](https://andres-vidal.github.io/ppforest2/main/r/reference/cutpoint_mean_of_means.md)
  (default).

- stop:

  A stopping rule object. Default depends on mode:
  [`stop_pure_node()`](https://andres-vidal.github.io/ppforest2/main/r/reference/stop_pure_node.md)
  for classification, and
  `stop_any(stop_min_size(5), stop_min_variance(0.01))` for regression.

- binarize:

  A binarization strategy object. Default depends on mode:
  [`binarize_largest_gap()`](https://andres-vidal.github.io/ppforest2/main/r/reference/binarize_largest_gap.md)
  for classification, and
  [`binarize_disabled()`](https://andres-vidal.github.io/ppforest2/main/r/reference/binarize_disabled.md)
  for regression (regression's default grouping always yields a 2-group
  partition, so no binarization is needed).

- grouping:

  A grouping strategy object. Default depends on mode:
  [`grouping_by_label()`](https://andres-vidal.github.io/ppforest2/main/r/reference/grouping_by_label.md)
  for classification, and
  [`grouping_by_cutpoint()`](https://andres-vidal.github.io/ppforest2/main/r/reference/grouping_by_cutpoint.md)
  for regression.

- leaf:

  A leaf strategy object. Default depends on mode:
  [`leaf_majority_vote()`](https://andres-vidal.github.io/ppforest2/main/r/reference/leaf_majority_vote.md)
  for classification, and
  [`leaf_mean_response()`](https://andres-vidal.github.io/ppforest2/main/r/reference/leaf_mean_response.md)
  for regression.

## Value

A `pptr` model. Its S3 class vector is
`c("pptr_classification", "pptr", "ppmodel")` or
`c("pptr_regression", "pptr", "ppmodel")` depending on the mode.

## Details

Mode is taken from the `mode` argument when explicit, and otherwise
auto-detected from \`y\` (factor/character → classification, numeric →
regression). Pass `mode = "classification"` to force classification on
integer labels (e.g. binary 0/1), or `mode = "regression"` to assert
intent on numeric responses.

## See also

[`predict.pptr_classification`](https://andres-vidal.github.io/ppforest2/main/r/reference/predict.pptr_classification.md),
[`predict.pptr_regression`](https://andres-vidal.github.io/ppforest2/main/r/reference/predict.pptr_regression.md),
[`formula.ppmodel`](https://andres-vidal.github.io/ppforest2/main/r/reference/formula.ppmodel.md),
[`print.pptr`](https://andres-vidal.github.io/ppforest2/main/r/reference/print.pptr.md),
[`save_json`](https://andres-vidal.github.io/ppforest2/main/r/reference/save_json.md),
[`load_json`](https://andres-vidal.github.io/ppforest2/main/r/reference/load_json.md),
[`pp_tree`](https://andres-vidal.github.io/ppforest2/main/r/reference/pp_tree.md)
for parsnip integration

## Examples

``` r
# Example 1: formula interface with the `iris` dataset
pptr(Species ~ ., data = iris)
#> 
#> Call: pptr(formula = Species ~ ., data = iris)
#> 
#> Project-Pursuit Oblique Decision Tree:
#> If ([ 0.01 0.04 -0.04 -0.01 ] * x) < 0.06660754:
#>  If ([ 0.04 0.07 -0.09 -0.15 ] * x) < -0.2075133:
#>    Predict: virginica 
#>  Else:
#>    Predict: versicolor 
#> Else:
#>   Predict: setosa 
#> 

# Example 2: formula interface with the `iris` dataset with regularization
pptr(Species ~ ., data = iris, lambda = 0.5)
#> 
#> Call: pptr(formula = Species ~ ., data = iris, lambda = 0.5)
#> 
#> Project-Pursuit Oblique Decision Tree:
#> If ([ 0 -0.04 0.03 0.03 ] * x) < 0.01580044:
#>   Predict: setosa 
#> Else:
#>  If ([ 0 0.03 -0.06 -0.15 ] * x) < -0.4503323:
#>    Predict: virginica 
#>  Else:
#>    Predict: versicolor 
#> 

# Example 3: matrix interface with the `iris` dataset
pptr(x = iris[, 1:4], y = iris[, 5])
#> 
#> Call: pptr(x = iris[, 1:4], y = iris[, 5])
#> 
#> Project-Pursuit Oblique Decision Tree:
#> If ([ 0.01 0.04 -0.04 -0.01 ] * x) < 0.06660754:
#>  If ([ 0.04 0.07 -0.09 -0.15 ] * x) < -0.2075133:
#>    Predict: virginica 
#>  Else:
#>    Predict: versicolor 
#> Else:
#>   Predict: setosa 
#> 
```
