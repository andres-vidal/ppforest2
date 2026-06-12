Sys.setenv(R_TESTS = "")
Sys.setenv(OMP_THREAD_LIMIT = "1")
Sys.setenv(OMP_NUM_THREADS = "1")

library(testthat)
library(ppforest2)

skip_if_not_installed("parsnip")
skip_if_not_installed("workflows")
skip_if_not_installed("rsample")
skip_if_not_installed("tune")
skip_if_not_installed("yardstick")

library(parsnip)
library(workflows)
library(rsample)
library(tune)

# Regression suite for the "object '..y' not found" bug. The bug surfaced
# whenever a `pp_rand_forest` / `pp_tree` spec was used through workflows
# (or anything downstream of it: `fit_resamples`, `tune_grid`). Root cause
# was that `set_fit(..., interface = "formula")` triggered parsnip's
# `xy_form()` dispatch when callers passed x/y data, which synthesises a
# `..y ~ ...` formula. `pprf()` stored that formula on the model, and
# prediction blew up because new data has no `..y` column. Switching the
# registration to `interface = "matrix"` made parsnip pass x/y straight
# through. These tests lock that wiring in.

# The ..y bug surfaced as silent per-fold fit *failures*, which tune records as
# error-type notes; assert there are none of those. Degenerate-split *warnings*
# (legitimate on small resampling folds, and more frequent now that the forest
# default subsamples features) are warning-type notes and must not fail this
# plumbing test.
expect_no_fit_errors <- function(rs, label) {
  expect_equal(sum(collect_notes(rs)$type == "error"), 0L, info = label)
}

describe("parsnip + workflows integration (..y bug regression)", {
  data(crabs, package = "ppforest2")

  make_classification_split <- function(seed = 0, prop = 0.8) {
    set.seed(seed)
    initial_split(crabs, prop = prop, strata = Type)
  }

  describe("pp_rand_forest", {
    it("fits and predicts through workflow() + add_formula()", {
      split <- make_classification_split()
      train <- training(split); test <- testing(split)
      spec <- pp_rand_forest(trees = 5, mtry = 2, penalty = 0.5) |>
        set_engine("ppforest2") |> set_mode("classification")
      wf <- workflow() |> add_model(spec) |> add_formula(Type ~ .)

      fit_wf <- fit(wf, data = train)
      expect_s3_class(fit_wf, "workflow")
      preds <- predict(fit_wf, new_data = test)
      expect_s3_class(preds, "tbl_df")
      expect_equal(nrow(preds), nrow(test))
      expect_true(".pred_class" %in% colnames(preds))
      expect_s3_class(preds$.pred_class, "factor")
    })

    it("fits and predicts through workflow() + add_variables()", {
      split <- make_classification_split()
      train <- training(split); test <- testing(split)
      spec <- pp_rand_forest(trees = 5, mtry = 2, penalty = 0.5) |>
        set_engine("ppforest2") |> set_mode("classification")
      wf <- workflow() |> add_model(spec) |>
        add_variables(outcomes = Type,
                      predictors = c(FL, RW, CL, CW, BD))

      fit_wf <- fit(wf, data = train)
      preds <- predict(fit_wf, new_data = test)
      expect_equal(nrow(preds), nrow(test))
      expect_true(".pred_class" %in% colnames(preds))
    })

    it("runs end-to-end through fit_resamples()", {
      split <- make_classification_split()
      train <- training(split)
      spec <- pp_rand_forest(trees = 5, mtry = 2, penalty = 0.5) |>
        set_engine("ppforest2") |> set_mode("classification")
      wf <- workflow() |> add_model(spec) |> add_formula(Type ~ .)
      folds <- vfold_cv(train, v = 3, strata = Type)

      rs <- fit_resamples(wf, resamples = folds)
      expect_no_fit_errors(rs, "fit_resamples produced fit-error notes")
      metrics <- collect_metrics(rs)
      expect_true("accuracy" %in% metrics$.metric)
      expect_equal(metrics$n[metrics$.metric == "accuracy"], 3L)
    })

    it("runs end-to-end through tune_grid()", {
      split <- make_classification_split()
      train <- training(split)
      spec <- pp_rand_forest(trees = 5, mtry = tune(), penalty = tune()) |>
        set_engine("ppforest2") |> set_mode("classification")
      wf <- workflow() |> add_model(spec) |> add_formula(Type ~ .)
      folds <- vfold_cv(train, v = 3, strata = Type)
      grid <- tibble::tibble(mtry = c(2L, 3L), penalty = c(0.0, 0.5))

      rs <- tune_grid(wf, resamples = folds, grid = grid)
      expect_no_fit_errors(rs, "tune_grid produced fit-error notes")
      metrics <- collect_metrics(rs)
      expect_equal(length(unique(metrics$.config)), nrow(grid))
    })

    it("runs end-to-end through tune_grid() over mtry_prop", {
      split <- make_classification_split()
      train <- training(split)
      spec <- pp_rand_forest(trees = 5, mtry_prop = tune(), penalty = 0) |>
        set_engine("ppforest2") |> set_mode("classification")
      wf <- workflow() |> add_model(spec) |> add_formula(Type ~ .)
      folds <- vfold_cv(train, v = 3, strata = Type)
      grid <- tibble::tibble(mtry_prop = c(0.4, 0.8))

      rs <- tune_grid(wf, resamples = folds, grid = grid)
      expect_no_fit_errors(rs, "tune_grid over mtry_prop produced fit-error notes")
      metrics <- collect_metrics(rs)
      expect_equal(length(unique(metrics$.config)), nrow(grid))
    })
  })

  describe("pp_tree", {
    it("runs end-to-end through fit_resamples()", {
      split <- make_classification_split()
      train <- training(split)
      spec <- pp_tree(penalty = 0) |>
        set_engine("ppforest2") |> set_mode("classification")
      wf <- workflow() |> add_model(spec) |> add_formula(Type ~ .)
      folds <- vfold_cv(train, v = 3, strata = Type)

      rs <- fit_resamples(wf, resamples = folds)
      expect_no_fit_errors(rs, "fit_resamples produced fit-error notes")
      expect_true("accuracy" %in% collect_metrics(rs)$.metric)
    })

    it("runs end-to-end through tune_grid()", {
      split <- make_classification_split()
      train <- training(split)
      spec <- pp_tree(penalty = tune()) |>
        set_engine("ppforest2") |> set_mode("classification")
      wf <- workflow() |> add_model(spec) |> add_formula(Type ~ .)
      folds <- vfold_cv(train, v = 3, strata = Type)
      grid <- tibble::tibble(penalty = c(0.0, 0.5))

      rs <- tune_grid(wf, resamples = folds, grid = grid)
      expect_no_fit_errors(rs, "tune_grid produced fit-error notes")
      expect_equal(length(unique(collect_metrics(rs)$.config)), nrow(grid))
    })
  })

  describe("regression mode", {
    it("pp_rand_forest fits and predicts through workflow + fit_resamples()", {
      set.seed(0)
      df <- data.frame(x1 = runif(60), x2 = runif(60))
      df$y <- df$x1 + 0.5 * df$x2 + rnorm(60, sd = 0.1)
      spec <- pp_rand_forest(trees = 5, mode = "regression") |>
        set_engine("ppforest2")
      wf <- workflow() |> add_model(spec) |> add_formula(y ~ .)
      folds <- vfold_cv(df, v = 3)

      rs <- fit_resamples(wf, resamples = folds)
      expect_no_fit_errors(rs, "regression fit_resamples produced fit-error notes")
      metrics <- collect_metrics(rs)
      expect_true("rmse" %in% metrics$.metric)
    })
  })
})
