# Trains a Random Forest of Project-Pursuit oblique decision trees.

This function trains a Random Forest of Project-Pursuit oblique decision
tree using either a formula and data frame interface or a matrix-based
interface. When using the formula interface, specify the model formula
and the data frame containing the variables. For the matrix-based
interface, provide matrices for the features and labels directly. The
number of trees is controlled by the `size` parameter. Each tree is
trained on a stratified bootstrap sample drawn from the data. The number
of variables to consider at each split is controlled by the `n_vars`
parameter. If `lambda = 0`, the model is trained using Linear
Discriminant Analysis (LDA). If `lambda > 0`, the model is trained using
Penalized Discriminant Analysis (PDA).

## Usage

``` r
pprf(
  formula = NULL,
  data = NULL,
  x = NULL,
  y = NULL,
  mode = NULL,
  size = 2,
  lambda = 0,
  n_vars = NULL,
  p_vars = NULL,
  seed = NULL,
  max_retries = 3L,
  threads = NULL,
  pp = NULL,
  vars = NULL,
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

- size:

  The number of trees in the forest.

- lambda:

  A regularization parameter. If `lambda = 0`, the model is trained
  using Linear Discriminant Analysis (LDA). If `lambda > 0`, the model
  is trained using Penalized Discriminant Analysis (PDA). Cannot be used
  together with `pp`.

- n_vars:

  The number of variables to consider at each split (integer). These are
  chosen uniformly in each split. The default is all variables. Cannot
  be used together with `p_vars` or `dr`.

- p_vars:

  The proportion of variables to consider at each split (number between
  0 and 1, exclusive). For example, `p_vars = 0.5` uses half the
  features. Cannot be used together with `n_vars` or `dr`.

- seed:

  An optional integer seed for reproducibility. If `NULL` (default), a
  seed is drawn from R's RNG, so
  [`set.seed()`](https://rdrr.io/r/base/Random.html) controls
  reproducibility. If an integer is provided, that value is used
  directly. The same seed is used for training and for computing
  permuted variable importance.

- max_retries:

  Maximum number of retries for degenerate trees (default: 3). When a
  bootstrap sample yields a singular covariance matrix, the tree is
  retrained with a different seed up to this many times.

- threads:

  The number of threads to use. The default is the number of cores
  available.

- pp:

  A projection pursuit strategy object created by
  [`pp_pda`](https://andres-vidal.github.io/ppforest2/main/r/reference/pp_pda.md).
  Cannot be used together with `lambda`.

- vars:

  A variable selection strategy object created by
  [`vars_uniform`](https://andres-vidal.github.io/ppforest2/main/r/reference/vars_uniform.md)
  or
  [`vars_all`](https://andres-vidal.github.io/ppforest2/main/r/reference/vars_all.md).
  Cannot be used together with `n_vars` or `p_vars`.

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

A `pprf` model. Its S3 class vector is
`c("pprf_classification", "pprf", "ppmodel")` or
`c("pprf_regression", "pprf", "ppmodel")` depending on the mode.

## Details

Mode is taken from the `mode` argument when explicit, and otherwise
auto-detected from \`y\` (factor/character → classification, numeric →
regression). Pass `mode = "classification"` to force classification on
integer labels (e.g. binary 0/1), or `mode = "regression"` to assert
intent on numeric responses.

OOB error, OOB predictions, permuted variable importance, and weighted
variable importance are computed lazily on first access via the accessor
functions (\`oob_error()\`, \`oob_predictions()\`,
\`permuted_importance()\`, \`weighted_importance()\`). Training itself
is fast because these OOB-based computations are deferred.

## See also

[`predict.pprf_classification`](https://andres-vidal.github.io/ppforest2/main/r/reference/predict.pprf_classification.md),
[`predict.pprf_regression`](https://andres-vidal.github.io/ppforest2/main/r/reference/predict.pprf_regression.md),
[`formula.ppmodel`](https://andres-vidal.github.io/ppforest2/main/r/reference/formula.ppmodel.md),
[`oob_error`](https://andres-vidal.github.io/ppforest2/main/r/reference/oob_error.md),
[`save_json`](https://andres-vidal.github.io/ppforest2/main/r/reference/save_json.md),
[`load_json`](https://andres-vidal.github.io/ppforest2/main/r/reference/load_json.md),
[`pp_rand_forest`](https://andres-vidal.github.io/ppforest2/main/r/reference/pp_rand_forest.md)
for parsnip integration,
[`vignette("introduction")`](https://andres-vidal.github.io/ppforest2/main/r/articles/introduction.md)
for a tutorial

## Examples

``` r
# Example 1: formula interface with the `iris` dataset
pprf(Species ~ ., data = iris)
#> 
#> Random Forest of Project-Pursuit Oblique Decision Trees
#>   Call:        pprf(formula = Species ~ ., data = iris)
#>   Trees:       2
#>   Mode:        classification
#>   Group names: setosa, versicolor, virginica
#>   Formula:     Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width -     1
#> 

# Example 2: formula interface with the `iris` dataset with regularization
pprf(Species ~ ., data = iris, lambda = 0.5)
#> 
#> Random Forest of Project-Pursuit Oblique Decision Trees
#>   Call:        pprf(formula = Species ~ ., data = iris, lambda = 0.5)
#>   Trees:       2
#>   Mode:        classification
#>   Group names: setosa, versicolor, virginica
#>   Formula:     Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width -     1
#> 

# Example 3: matrix interface with the `iris` dataset
pprf(x = iris[, 1:4], y = iris[, 5])
#> 
#> Random Forest of Project-Pursuit Oblique Decision Trees
#>   Call:        pprf(x = iris[, 1:4], y = iris[, 5])
#>   Trees:       2
#>   Mode:        classification
#>   Group names: setosa, versicolor, virginica
#> 

# Example 4: matrix interface with the `iris` dataset with regularization
pprf(x = iris[, 1:4], y = iris[, 5], lambda = 0.5)
#> 
#> Random Forest of Project-Pursuit Oblique Decision Trees
#>   Call:        pprf(x = iris[, 1:4], y = iris[, 5], lambda = 0.5)
#>   Trees:       2
#>   Mode:        classification
#>   Group names: setosa, versicolor, virginica
#> 

# Example 5: formula interface with the `crabs` dataset
pprf(Type ~ ., data = crabs)
#> 
#> Random Forest of Project-Pursuit Oblique Decision Trees
#>   Call:        pprf(formula = Type ~ ., data = crabs)
#>   Trees:       2
#>   Mode:        classification
#>   Group names: B, O
#>   Formula:     Type ~ sex + FL + RW + CL + CW + BD - 1
#> 

# Example 6: formula interface with the `crabs` dataset with regularization
pprf(Type ~ ., data = crabs, lambda = 0.5)
#> 
#> Random Forest of Project-Pursuit Oblique Decision Trees
#>   Call:        pprf(formula = Type ~ ., data = crabs, lambda = 0.5)
#>   Trees:       2
#>   Mode:        classification
#>   Group names: B, O
#>   Formula:     Type ~ sex + FL + RW + CL + CW + BD - 1
#> 
```
