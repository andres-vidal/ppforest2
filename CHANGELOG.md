# ppforest2 0.1.0

## New features

- Core: Projection-pursuit oblique decision trees and random forests for classification, using LDA/PDA optimization.
- Core: Random uniform variable selection per split for forest diversity.
- Core: Three variable importance measures: permuted (VI1), projections (VI2), and weighted projections (VI3).
- Core: Out-of-bag error and confusion matrix for forests, with bootstrap sample indices persisted for recomputation. `oob_error` is `NA_real_` (R) / "not available" (CLI) when no observation has any out-of-bag tree.
- Core: Degenerate split detection when projection-pursuit cannot find a useful projection.
- Core: OpenMP multi-threaded forest training.
- Core: JSON serialization for trained models. Optional metrics fields use a uniform `null`-or-value representation so downstream tooling can distinguish "computed but empty" from other shapes without special-casing.
- Core: Cross-platform reproducibility — identical results for the same seed on Linux, macOS, and Windows, enforced by golden-file tests in CI.
- R: `pptr()` and `pprf()` with formula and matrix interfaces. Returned models carry an S3 class vector identifying both model type and mode (e.g. `c("pprf_classification", "pprf", "ppmodel")`).
- R: `predict()` returns group labels (`type = "class"`) or vote proportions (`type = "prob"`) for classification.
- R: `summary()` displays training and OOB confusion matrices.
- R: Lazy OOB accessors — `oob_error()`, `oob_predictions()`, `oob_samples()`, `bag_samples()`, `permuted_importance()`, `weighted_importance()` — compute from the training data stored on the model on first access and memoize in an environment cache, so training is fast and repeated access is free. `oob_predictions()` returns a factor with `NA` for rows with no OOB tree.
- R: Permuted variable importance may be negative; this is meaningful signal ("within noise") rather than a sentinel, so callers should rely on the ranking rather than clipping at zero. Weighted projection importance is non-negative by construction.
- R: Degenerate split warnings when projection-pursuit cannot separate groups.
- R: `save_json()` and `load_json()` for model persistence.
- R: tidymodels/parsnip integration: `pp_tree()` and `pp_rand_forest()` model specifications.
- R: ggplot2 visualizations — tree diagrams, variable importance plots, projection histograms, and decision boundary plots.
- R: Bundled classification datasets: crab, crabs, fishcatch, glass, image, leukemia, lymphoma, NCI60, olive, parkinson, and wine. (Use `datasets::iris` from base R for iris examples.)
- CLI: `train` fits a tree or forest from CSV and saves as JSON.
- CLI: `predict` applies a saved model to new data.
- CLI: `evaluate` runs train/test evaluation with smart convergence.
- CLI: `summarize` displays model configuration, data summary, and metrics from a saved model JSON. `--data` recomputes metrics from training data.
- CLI: `benchmark` runs multi-scenario performance benchmarks with baseline comparison.

## Experimental features

Regression support is included but untested in production workloads. API surface and defaults may change in future releases.

- Core: Regression training via a `ByCutpoint` grouping strategy that quantile-slices the continuous response, a `MeanResponse` leaf, and `MinSize` / `MinVariance` / `CompositeStop` (`stop::any`) stop rules.
- Core: Regression metrics — MSE, MAE, and R² — computed for training and out-of-bag predictions. Forest OOB error is reported as MSE.
- R: Regression auto-detected when `y` is numeric (not a factor). `predict()` returns a numeric vector (`type = "response"`).
- R: Regression strategy wrappers: `grouping_by_cutpoint()`, `leaf_mean_response()`, `stop_min_size()`, `stop_min_variance()`, `stop_any()`.
- R: `summary()` displays MSE / MAE / R² for regression models. `oob_predictions()` returns a numeric vector with `NA_real_` for rows with no OOB tree.
- R: `save_json()` / `load_json()` preserve regression mode; parsnip `pp_tree()` / `pp_rand_forest()` accept `mode = "regression"`.
- CLI: `--mode classification|regression` selects the training mode; regression reads the last CSV column as the continuous response. `predict` returns numeric predictions and MSE/MAE/R²; `evaluate` reports MSE for regression.
- R: Bundled regression dataset `california_housing` (20,433 × 9, predict `median_house_value`). For smaller regression examples use `datasets::mtcars` from base R.
