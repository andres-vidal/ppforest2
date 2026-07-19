# Changelog

## ppforest2 0.1.1

CRAN release: 2026-07-19

### Packaging and build

- R: The R package now compiles the C++ core directly through `Makevars`
  instead of CMake, with no network access or downloaded dependencies at
  install time. Eigen is provided by RcppEigen; nlohmann/json and pcg
  headers are vendored under `inst/include`. This makes the package
  installable on CRAN’s offline build machines. (`fmt` and `csv-parser`,
  used only by the CLI, are no longer part of the R build.)
- R: Self-registering strategies are compiled directly into the shared
  object, removing the previous whole-archive linking workaround.
- R: Compiler flags for the direct build mirror the standalone C++ build
  — `EIGEN_NO_AUTOMATIC_RESIZING` on all platforms and
  `EIGEN_DONT_VECTORIZE` on Windows.
- Core: Replaced C++20 designated initializers and fixed
  member-initialization order in `stats/GroupPartition` and
  `stats/Simulation` so the code compiles warning-free under a strict
  C++17 GCC (`-Wall -Wextra -pedantic`).
- Core: A compile-time `EIGEN_VERSION_AT_LEAST(3, 4, 0)` guard fails the
  build with a clear message if an incompatible Eigen is supplied
  (e.g. via RcppEigen).
- `make r-vendor-deps` re-vendors the committed json/pcg headers after a
  version bump in `core/Dependencies.cmake`.

### Documentation

- R: `DESCRIPTION` uses `Authors@R` and cites the projection-pursuit
  tree and forest references with DOIs.
- R: Examples for the parsnip and plot methods use `\donttest` with
  [`requireNamespace()`](https://rdrr.io/r/base/ns-load.html) guards
  instead of `\dontrun`, so they run under `--run-donttest` when the
  suggested packages are available.

### CRAN

- Added `cran-comments.md`. The package passes `R CMD check --as-cran`
  with no errors or warnings; remaining notes (new submission, cosmetic
  pragmas in the vendored nlohmann/json headers) are documented for the
  reviewer.

## ppforest2 0.1.0

### New features

- Core: Projection-pursuit oblique decision trees and random forests for
  classification, using LDA/PDA optimization.
- Core: Random uniform variable selection per split for forest
  diversity.
- Core: Three variable importance measures: permuted (VI1), projections
  (VI2), and weighted projections (VI3).
- Core: Out-of-bag error and confusion matrix for forests, with
  bootstrap sample indices persisted for recomputation. `oob_error` is
  `NA_real_` (R) / “not available” (CLI) when no observation has any
  out-of-bag tree.
- Core: Degenerate split detection when projection-pursuit cannot find a
  useful projection.
- Core: OpenMP multi-threaded forest training.
- Core: JSON serialization for trained models. Optional metrics fields
  use a uniform `null`-or-value representation so downstream tooling can
  distinguish “computed but empty” from other shapes without
  special-casing.
- Core: Cross-platform reproducibility — identical results for the same
  seed on Linux, macOS, and Windows, enforced by golden-file tests in
  CI.
- R:
  [`pptr()`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/pptr.md)
  and
  [`pprf()`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/pprf.md)
  with formula and matrix interfaces. Returned models carry an S3 class
  vector identifying both model type and mode
  (e.g. `c("pprf_classification", "pprf", "ppmodel")`).
- R: [`predict()`](https://rdrr.io/r/stats/predict.html) returns group
  labels (`type = "class"`) or vote proportions (`type = "prob"`) for
  classification.
- R: [`summary()`](https://rdrr.io/r/base/summary.html) displays
  training and OOB confusion matrices.
- R: Lazy OOB accessors —
  [`oob_error()`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/oob_error.md),
  [`oob_predictions()`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/oob_predictions.md),
  [`oob_samples()`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/oob_samples.md),
  [`bag_samples()`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/bag_samples.md),
  [`permuted_importance()`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/permuted_importance.md),
  [`weighted_importance()`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/weighted_importance.md)
  — compute from the training data stored on the model on first access
  and memoize in an environment cache, so training is fast and repeated
  access is free.
  [`oob_predictions()`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/oob_predictions.md)
  returns a factor with `NA` for rows with no OOB tree.
- R: Permuted variable importance may be negative; this is meaningful
  signal (“within noise”) rather than a sentinel, so callers should rely
  on the ranking rather than clipping at zero. Weighted projection
  importance is non-negative by construction.
- R: Degenerate split warnings when projection-pursuit cannot separate
  groups.
- R:
  [`save_json()`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/save_json.md)
  and
  [`load_json()`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/load_json.md)
  for model persistence.
- R: tidymodels/parsnip integration:
  [`pp_tree()`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/pp_tree.md)
  and
  [`pp_rand_forest()`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/pp_rand_forest.md)
  model specifications.
- R: ggplot2 visualizations — tree diagrams, variable importance plots,
  projection histograms, and decision boundary plots.
- R: Bundled classification datasets: crab, crabs, fishcatch, glass,
  image, leukemia, lymphoma, NCI60, olive, parkinson, and wine. (Use
  [`datasets::iris`](https://rdrr.io/r/datasets/iris.html) from base R
  for iris examples.)
- CLI: `train` fits a tree or forest from CSV and saves as JSON.
- CLI: `predict` applies a saved model to new data.
- CLI: `evaluate` runs train/test evaluation with smart convergence.
- CLI: `summarize` displays model configuration, data summary, and
  metrics from a saved model JSON. `--data` recomputes metrics from
  training data.
- CLI: `benchmark` runs multi-scenario performance benchmarks with
  baseline comparison.
- CLI: `serve` exposes a saved model over HTTP — `GET /` returns the
  model summary as JSON (or an HTML dashboard for browsers showing
  configuration, training metrics, and variable importance),
  `GET /health` is a liveness probe, and `POST /predict` accepts a
  feature CSV and returns predictions (JSON for API clients, an HTML
  predictions page for browsers, with a Download CSV button and
  confusion matrix when the request CSV includes a response column).
  Results are cached in-memory with shareable `?id=…` URLs; the
  dashboard binds to `127.0.0.1:8080` by default.

### Experimental features

Regression support is included but untested in production workloads. API
surface and defaults may change in future releases.

- Core: Regression training via a `ByCutpoint` grouping strategy that
  quantile-slices the continuous response, a `MeanResponse` leaf, and
  `MinSize` / `MinVariance` / `CompositeStop` (`stop::any`) stop rules.
- Core: Regression metrics — MSE, MAE, and R² — computed for training
  and out-of-bag predictions. Forest OOB error is reported as MSE.
- R: Regression auto-detected when `y` is numeric (not a factor).
  [`predict()`](https://rdrr.io/r/stats/predict.html) returns a numeric
  vector (`type = "response"`).
- R: Regression strategy wrappers:
  [`grouping_by_cutpoint()`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/grouping_by_cutpoint.md),
  [`leaf_mean_response()`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/leaf_mean_response.md),
  [`stop_min_size()`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/stop_min_size.md),
  [`stop_min_variance()`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/stop_min_variance.md),
  [`stop_any()`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/stop_any.md).
- R: [`summary()`](https://rdrr.io/r/base/summary.html) displays MSE /
  MAE / R² for regression models.
  [`oob_predictions()`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/oob_predictions.md)
  returns a numeric vector with `NA_real_` for rows with no OOB tree.
- R:
  [`save_json()`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/save_json.md)
  /
  [`load_json()`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/load_json.md)
  preserve regression mode; parsnip
  [`pp_tree()`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/pp_tree.md)
  /
  [`pp_rand_forest()`](https://andres-vidal.github.io/ppforest2/v0.1.1/r/reference/pp_rand_forest.md)
  accept `mode = "regression"`.
- CLI: `--mode classification|regression` selects the training mode;
  regression reads the last CSV column as the continuous response.
  `predict` returns numeric predictions and MSE/MAE/R²; `evaluate`
  reports MSE for regression.
- R: Bundled regression dataset `california_housing` (20,433 × 9,
  predict `median_house_value`). For smaller regression examples use
  [`datasets::mtcars`](https://rdrr.io/r/datasets/mtcars.html) from base
  R.
