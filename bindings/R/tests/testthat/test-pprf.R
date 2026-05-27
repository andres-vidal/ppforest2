Sys.setenv(R_TESTS = "")
Sys.setenv(OMP_THREAD_LIMIT = "1")
Sys.setenv(OMP_NUM_THREADS = "1")

library(testthat)
library(ppforest2)

describe("pprf formula interface", {
  it("returns a object with s3 class pprf", {
    model <- pprf(Species ~ ., data = iris, threads = 1)
    expect_s3_class(model, "pprf")
  })

  it("preserves the observations (x) as a matrix in the returned model", {
    model <- pprf(Species ~ ., data = iris, threads = 1)
    expected <- model.matrix(
      Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width - 1,
      iris,
      response = TRUE
    )
    expect_equal(model$x, expected)
  })

  it("preserves the labels (y) as a matrix in the returned model", {
    model <- pprf(Species ~ ., data = iris, threads = 1)
    expected <- as.matrix(as.numeric(as.factor(iris$Species)))
    expect_equal(model$y, expected)
  })

  it("preserves the groups in the returned model", {
    model <- pprf(Species ~ ., data = iris, threads = 1)
    expected <- levels(iris$Species)
    expect_equal(model$groups, expected)
  })

  it("preserves the formula in the returned model", {
    model <- pprf(Species ~ ., data = iris, threads = 1)

    expect_equal(
      model$formula,
      Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width - 1
    )
  })
})

describe("pprf matrix interface", {
  it("returns a object with s3 class pprf", {
    model <- pprf(x = iris[, 1:4], y = iris[, 5], threads = 1)
    expect_s3_class(model, "pprf")
  })

  it("preserves the observations (x) as a matrix in the returned model", {
    model <- pprf(x = iris[, 1:4], y = iris[, 5], threads = 1)
    expected <- as.matrix(iris[, 1:4])
    expect_equal(model$x, expected)
  })

  it("preserves the labels (y) as a matrix in the returned model", {
    model <- pprf(x = iris[, 1:4], y = iris[, 5], threads = 1)
    expected <- as.matrix(as.numeric(as.factor(iris$Species)))
    expect_equal(model$y, expected)
  })

  it("preserves the groups in the returned model", {
    model <- pprf(x = iris[, 1:4], y = iris[, 5], threads = 1)
    expected <- levels(iris$Species)
    expect_equal(model$groups, expected)
  })

  it("does not have a formula", {
    model <- pprf(x = iris[, 1:4], y = iris[, 5], threads = 1)
    expect_equal(model$formula, NULL)
  })
})

describe("pprf reproducibility", {
  it("produces identical predictions with set.seed", {
    set.seed(0)
    model1 <- pprf(Species ~ ., data = iris, size = 3, threads = 1)
    set.seed(0)
    model2 <- pprf(Species ~ ., data = iris, size = 3, threads = 1)
    expect_equal(predict(model1, iris), predict(model2, iris))
  })

  it("produces identical predictions with explicit seed", {
    model1 <- pprf(Species ~ ., data = iris, size = 3, seed = 123L, threads = 1)
    model2 <- pprf(Species ~ ., data = iris, size = 3, seed = 123L, threads = 1)
    expect_equal(predict(model1, iris), predict(model2, iris))
  })

  it("produces reproducible VI permuted importance", {
    model1 <- pprf(Species ~ ., data = iris, size = 3, seed = 0, threads = 1)
    model2 <- pprf(Species ~ ., data = iris, size = 3, seed = 0, threads = 1)
    expect_equal(permuted_importance(model1), permuted_importance(model2))
  })

  it("stores the explicit seed in the model", {
    model <- pprf(Species ~ ., data = iris, size = 3, seed = 0, threads = 1)
    expect_equal(model$seed, 0)
  })

  it("stores the generated seed when seed is NULL", {
    set.seed(99)
    model <- pprf(Species ~ ., data = iris, size = 3, threads = 1)
    expect_true(is.numeric(model$seed))
    expect_true(model$seed > 0)
  })
})

describe("pprf input validation", {
  it("rejects invalid lambda", {
    expect_error(pprf(x = iris[, 1:4], y = iris[, 5], lambda = -1), "between 0 and 1")
    expect_error(pprf(x = iris[, 1:4], y = iris[, 5], lambda = 2), "between 0 and 1")
  })

  it("rejects non-integer seed", {
    expect_error(pprf(x = iris[, 1:4], y = iris[, 5], seed = 1.5), "integer")
  })

  it("rejects size < 1", {
    expect_error(pprf(x = iris[, 1:4], y = iris[, 5], size = 0), "positive integer")
  })

  it("rejects non-integer size", {
    expect_error(pprf(x = iris[, 1:4], y = iris[, 5], size = 2.5), "positive integer")
  })

  it("rejects n_vars out of range", {
    expect_error(pprf(x = iris[, 1:4], y = iris[, 5], n_vars = 100), "number of features")
    expect_error(pprf(x = iris[, 1:4], y = iris[, 5], n_vars = 0), "positive integer")
  })

  it("rejects non-positive threads", {
    expect_error(pprf(x = iris[, 1:4], y = iris[, 5], threads = 0), "positive integer")
    expect_error(pprf(x = iris[, 1:4], y = iris[, 5], threads = -1), "positive integer")
  })

  it("rejects both n_vars and p_vars", {
    expect_error(pprf(x = iris[, 1:4], y = iris[, 5], n_vars = 2, p_vars = 0.5), "not both")
  })

  it("rejects invalid p_vars", {
    expect_error(pprf(x = iris[, 1:4], y = iris[, 5], p_vars = 0), "between 0 (exclusive) and 1 (inclusive)", fixed = TRUE)
    expect_error(pprf(x = iris[, 1:4], y = iris[, 5], p_vars = 1.5), "between 0 (exclusive) and 1 (inclusive)", fixed = TRUE)
  })

  it("accepts valid p_vars", {
    model <- pprf(x = iris[, 1:4], y = iris[, 5], p_vars = 0.5, size = 2, threads = 1)
    expect_equal(model$training_spec$vars$count, 2)
  })
})

describe("pprf training spec", {
  it("preserves the lambda parameter in the returned model", {
    model <- pprf(Species ~ ., data = iris, lambda = 0.5, threads = 1)
    expect_equal(model$training_spec$pp$lambda, 0.5)
  })

  it("the lambda parameter is 0 by default", {
    model <- pprf(Species ~ ., data = iris, threads = 1)
    expect_equal(model$training_spec$pp$lambda, 0)
  })

  it("preserves the n_vars parameter in the returned model", {
    model <- pprf(Species ~ ., data = iris, n_vars = 2, threads = 1)
    expect_equal(model$training_spec$vars$count, 2)
  })

  it("the n_vars parameter is the number of variables by default", {
    model <- pprf(x = iris[, 1:4], y = iris[, 5], threads = 1)
    expect_equal(model$training_spec$vars$count, ncol(iris) - 1)
  })

  it("generates as many trees as indicated by the size parameter", {
    model <- pprf(Species ~ ., data = iris, size = 3, threads = 1)
    expect_equal(length(model$trees), 3)
  })

  it("generates more than one tree if the size parameter is not given", {
    model <- pprf(Species ~ ., data = iris, threads = 1)
    expect_true(length(model$trees) > 1)
  })

  it("the pp strategy is pda", {
    model <- pprf(Species ~ ., data = iris, threads = 1)
    expect_equal(model$training_spec$pp$name, "pda")
  })

  it("the vars strategy is uniform", {
    model <- pprf(Species ~ ., data = iris, threads = 1)
    expect_equal(model$training_spec$vars$name, "uniform")
  })

  it("the stop strategy is pure_node", {
    model <- pprf(Species ~ ., data = iris, threads = 1)
    expect_equal(model$training_spec$stop$name, "pure_node")
  })

  it("the binarize strategy is largest_gap", {
    model <- pprf(Species ~ ., data = iris, threads = 1)
    expect_equal(model$training_spec$binarize$name, "largest_gap")
  })

  it("the grouping strategy is by_label", {
    model <- pprf(Species ~ ., data = iris, threads = 1)
    expect_equal(model$training_spec$grouping$name, "by_label")
  })

  it("the leaf strategy is majority_vote", {
    model <- pprf(Species ~ ., data = iris, threads = 1)
    expect_equal(model$training_spec$leaf$name, "majority_vote")
  })
})

describe("pprf regression training spec", {
  # Pins the mode-aware defaults selected by
  # `TrainingSpec::Builder::apply_defaults()` for regression. The C++
  # side is the single source of truth; this block guards against
  # silent drift if the default selection ever changes.
  it("the pp strategy is pda", {
    d <- make_regression_data()
    model <- pprf(x = d$x, y = d$y, seed = 0, threads = 1)
    expect_equal(model$training_spec$pp$name, "pda")
  })

  it("the vars strategy is uniform (forest default)", {
    # Unlike trees, forests subsample variables by default — same
    # convention as the classification path.
    d <- make_regression_data()
    model <- pprf(x = d$x, y = d$y, seed = 0, threads = 1)
    expect_equal(model$training_spec$vars$name, "uniform")
  })

  it("the cutpoint strategy is mean_of_means", {
    d <- make_regression_data()
    model <- pprf(x = d$x, y = d$y, seed = 0, threads = 1)
    expect_equal(model$training_spec$cutpoint$name, "mean_of_means")
  })

  it("the stop strategy is any(min_size(5), min_variance(0.01))", {
    d <- make_regression_data()
    model <- pprf(x = d$x, y = d$y, seed = 0, threads = 1)
    stop_spec <- model$training_spec$stop
    expect_equal(stop_spec$name, "any")
    expect_length(stop_spec$rules, 2)
    expect_equal(stop_spec$rules[[1]]$name, "min_size")
    expect_equal(stop_spec$rules[[1]]$min_size, 5)
    expect_equal(stop_spec$rules[[2]]$name, "min_variance")
    expect_equal(stop_spec$rules[[2]]$threshold, 0.01, tolerance = 1e-6)
  })

  it("the binarize strategy is disabled", {
    # Regression's `by_cutpoint` grouping always yields a 2-group
    # partition, so binarize is a no-op placeholder.
    d <- make_regression_data()
    model <- pprf(x = d$x, y = d$y, seed = 0, threads = 1)
    expect_equal(model$training_spec$binarize$name, "disabled")
  })

  it("the grouping strategy is by_cutpoint", {
    d <- make_regression_data()
    model <- pprf(x = d$x, y = d$y, seed = 0, threads = 1)
    expect_equal(model$training_spec$grouping$name, "by_cutpoint")
  })

  it("the leaf strategy is mean_response", {
    d <- make_regression_data()
    model <- pprf(x = d$x, y = d$y, seed = 0, threads = 1)
    expect_equal(model$training_spec$leaf$name, "mean_response")
  })
})

describe("pprf with strategy objects", {
  it("trains with pp and vars", {
    model <- pprf(Species ~ ., data = iris, size = 2, pp = pp_pda(0.5), vars = vars_uniform(n_vars = 2), seed = 0, threads = 1)
    expect_s3_class(model, "pprf")
    expect_equal(model$training_spec$pp$lambda, 0.5)
    expect_equal(model$training_spec$vars$count, 2)
  })

  it("trains with vars using p_vars", {
    model <- pprf(Species ~ ., data = iris, size = 2, vars = vars_uniform(p_vars = 0.5), seed = 0, threads = 1)
    expect_s3_class(model, "pprf")
    expect_equal(model$training_spec$vars$count, 2)  # 0.5 * 4 = 2
  })

  it("produces identical export as shortcut params", {
    model_shortcut <- pprf(Species ~ ., data = iris, size = 3, lambda = 0.5, n_vars = 2, seed = 0, threads = 1)
    model_strategy <- pprf(Species ~ ., data = iris, size = 3, pp = pp_pda(0.5), vars = vars_uniform(n_vars = 2), seed = 0, threads = 1)

    path_shortcut <- tempfile(fileext = ".json")
    path_strategy <- tempfile(fileext = ".json")
    save_json(model_shortcut, path_shortcut)
    save_json(model_strategy, path_strategy)

    j_shortcut <- jsonlite::fromJSON(readLines(path_shortcut, warn = FALSE))
    j_strategy <- jsonlite::fromJSON(readLines(path_strategy, warn = FALSE))

    j_shortcut$training_duration_ms <- NULL
    j_strategy$training_duration_ms <- NULL
    expect_equal(j_shortcut, j_strategy)
  })

  it("p_vars shortcut and vars_uniform(p_vars) produce identical export", {
    model_shortcut <- pprf(Species ~ ., data = iris, size = 3, p_vars = 0.5, seed = 0, threads = 1)
    model_strategy <- pprf(Species ~ ., data = iris, size = 3, vars = vars_uniform(p_vars = 0.5), seed = 0, threads = 1)

    path_shortcut <- tempfile(fileext = ".json")
    path_strategy <- tempfile(fileext = ".json")
    save_json(model_shortcut, path_shortcut)
    save_json(model_strategy, path_strategy)

    j_shortcut <- jsonlite::fromJSON(readLines(path_shortcut, warn = FALSE))
    j_strategy <- jsonlite::fromJSON(readLines(path_strategy, warn = FALSE))

    j_shortcut$training_duration_ms <- NULL
    j_strategy$training_duration_ms <- NULL
    expect_equal(j_shortcut, j_strategy)
  })

  it("errors when mixing pp and lambda", {
    expect_error(
      pprf(Species ~ ., data = iris, pp = pp_pda(0.5), lambda = 0.3),
      "Cannot use `pp` together with `lambda`")
  })

  it("errors when mixing vars and n_vars", {
    expect_error(
      pprf(Species ~ ., data = iris, vars = vars_uniform(n_vars = 2), n_vars = 3),
      "Cannot use `vars` together with `n_vars`/`p_vars`")
  })

  it("errors when mixing vars and p_vars", {
    expect_error(
      pprf(Species ~ ., data = iris, vars = vars_uniform(n_vars = 2), p_vars = 0.5),
      "Cannot use `vars` together with `n_vars`/`p_vars`")
  })

  it("rejects non-vars_strategy objects for vars", {
    expect_error(
      pprf(Species ~ ., data = iris, vars = list(name = "uniform", count = 2)),
      "vars_strategy object")
  })

  it("works with parsnip engine args pattern", {
    model <- pprf(Species ~ ., data = iris, size = 2, pp = pp_pda(0.5), vars = vars_uniform(n_vars = 2), seed = 0, threads = 1)
    preds <- predict(model, iris)
    expect_length(preds, 150)
  })

  it("trains with vars_all (no variable subsampling)", {
    model <- pprf(Species ~ ., data = iris, size = 2, vars = vars_all(), seed = 0, threads = 1)
    expect_s3_class(model, "pprf")
    expect_equal(model$training_spec$vars$name, "all")
  })

  it("training spec reflects strategy objects, not just defaults", {
    model <- pprf(Species ~ ., data = iris, size = 2, pp = pp_pda(0.7), vars = vars_uniform(n_vars = 3), cutpoint = cutpoint_mean_of_means(), seed = 0, threads = 1)
    expect_equal(model$training_spec$pp$name, "pda")
    expect_equal(model$training_spec$pp$lambda, 0.7, tolerance = 1e-5)
    expect_equal(model$training_spec$vars$name, "uniform")
    expect_equal(model$training_spec$vars$count, 3)
    expect_equal(model$training_spec$cutpoint$name, "mean_of_means")
  })

  it("trains with explicit stop, binarize, grouping, leaf", {
    model <- pprf(Species ~ ., data = iris, size = 2, stop = stop_pure_node(), binarize = binarize_largest_gap(), grouping = grouping_by_label(), leaf = leaf_majority_vote(), seed = 0, threads = 1)
    expect_s3_class(model, "pprf")
    expect_equal(model$training_spec$stop$name, "pure_node")
    expect_equal(model$training_spec$binarize$name, "largest_gap")
    expect_equal(model$training_spec$grouping$name, "by_label")
    expect_equal(model$training_spec$leaf$name, "majority_vote")
  })

  it("rejects non-stop_strategy objects", {
    expect_error(
      pprf(Species ~ ., data = iris, stop = list(name = "pure_node")),
      "stop_strategy")
  })

  it("rejects non-binarize_strategy objects", {
    expect_error(
      pprf(Species ~ ., data = iris, binarize = list(name = "largest_gap")),
      "binarize_strategy")
  })

  it("rejects non-grouping_strategy objects", {
    expect_error(
      pprf(Species ~ ., data = iris, grouping = list(name = "by_label")),
      "grouping_strategy")
  })

  it("rejects non-leaf_strategy objects", {
    expect_error(
      pprf(Species ~ ., data = iris, leaf = list(name = "majority_vote")),
      "leaf_strategy")
  })
})

describe("pprf edge cases", {
  it("rejects NA in features via matrix interface", {
    x <- matrix(c(1, NA, 3, 4, 1, 2, 3, 4), ncol = 2)
    y <- c("a", "a", "b", "b")
    expect_error(pprf(x = x, y = y, size = 3, threads = 1), "NA or NaN")
  })

  it("rejects NaN in features via matrix interface", {
    x <- matrix(c(1, NaN, 3, 4, 1, 2, 3, 4), ncol = 2)
    y <- c("a", "a", "b", "b")
    expect_error(pprf(x = x, y = y, size = 3, threads = 1), "NA or NaN")
  })

  it("trains with a constant feature column", {
    df <- data.frame(x1 = rep(5, 6), x2 = c(1, 2, 3, 7, 8, 9), y = c("a", "a", "a", "b", "b", "b"))
    expect_no_error(pprf(y ~ ., data = df, size = 3, seed = 0, threads = 1))
  })

  it("trains with single observation per group", {
    df <- data.frame(x1 = c(1, 0), x2 = c(0, 1), y = c("a", "b"))
    model <- pprf(y ~ ., data = df, size = 3, seed = 0, threads = 1)
    expect_s3_class(model, "pprf")
    preds <- predict(model, df)
    expect_length(preds, 2)
  })

  it("trains with minimal dataset (n=2)", {
    df <- data.frame(x1 = c(1, 9), y = c("a", "b"))
    model <- pprf(y ~ ., data = df, size = 3, seed = 0, threads = 1)
    expect_s3_class(model, "pprf")
    preds <- predict(model, df)
    expect_length(preds, 2)
  })

  it("trains with extreme class imbalance", {
    df <- data.frame(
      x1 = c(0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 90, 91),
      x2 = c(0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 90, 91),
      y = c(rep("a", 18), rep("b", 2))
    )
    model <- pprf(y ~ ., data = df, size = 5, seed = 0, threads = 1)
    expect_s3_class(model, "pprf")
    preds <- predict(model, df)
    expect_length(preds, 20)
  })
})

describe("pprf regression", {
  it("end-to-end on mtcars (formula + predict + summary)", {
    # Real-dataset round-trip counterpart to the simulated-data tests
    # below. mtcars is small (32 x 11) and ships with base R so this
    # doesn't grow the package footprint, while still exercising the
    # full pipeline on real covariances: formula interface, regression
    # auto-detection from numeric `y`, training, prediction, and
    # summary rendering.
    data(mtcars)
    model <- pprf(mpg ~ ., data = mtcars, size = 10, seed = 0, threads = 1)

    expect_equal(model$mode, "regression")
    # `groups` is an empty character vector for regression (no class levels).
    expect_length(model$groups, 0)
    expect_s3_class(model, "pprf_regression")
    expect_s3_class(model, "pprf")
    expect_s3_class(model, "ppmodel")

    preds <- predict(model, newdata = mtcars)
    expect_true(is.numeric(preds))
    expect_length(preds, nrow(mtcars))
    expect_true(all(is.finite(preds)))

    # summary() must render the regression-shaped metrics block (MSE /
    # MAE / R²) and must not render a confusion matrix.
    out <- capture.output(summary(model))
    expect_true(any(grepl("Training", out)))
    expect_true(any(grepl("MSE|R\\^2|R\u00b2|MAE", out)))
    expect_false(any(grepl("Confusion Matrix", out)))

    # Sanity on OOB error: MSE is non-negative, or NA if no row has OOB.
    err <- oob_error(model)
    expect_true(is.na(err) || err >= 0)
  })

  it("auto-detects regression from numeric y", {
    d <- make_regression_data(n = 40)
    model <- pprf(x = d$x, y = d$y, size = 5, seed = 0, threads = 1)

    expect_equal(model$mode, "regression")
    expect_length(model$trees, 5)
  })

  it("predict returns numeric", {
    d <- make_regression_data(n = 40)
    model <- pprf(x = d$x, y = d$y, size = 5, seed = 0, threads = 1)
    preds <- predict(model, d$x)

    expect_type(preds, "double")
    expect_length(preds, 40)
  })

  it("produces a reasonable MSE", {
    d <- make_regression_data(n = 50)
    model <- pprf(x = d$x, y = d$y, size = 20, seed = 0, threads = 1)
    preds <- predict(model, d$x)
    mse <- mean((preds - d$y)^2)

    expect_lt(mse, 1.0)
  })

  it("oob_error is an MSE", {
    d <- make_regression_data(n = 40)
    model <- pprf(x = d$x, y = d$y, size = 10, seed = 0, threads = 1)

    expect_true(is.numeric(oob_error(model)))
    expect_gte(oob_error(model), 0)
  })

  it("oob_predictions uses NA_real_ (not NaN) for no-OOB rows", {
    # C++ emits NaN for rows that no tree left out-of-bag; the R boundary
    # remaps those to NA_real_ so the missing-value story is uniform with
    # the classification path (which returns NA at the factor level). This
    # test pins that remap: force every row to be in-bag in every tree by
    # rewriting `sample_indices` to cover `0:(n-1)`, then confirm
    #   (a) every entry is NA under `is.na()`, and
    #   (b) no entry is the raw C++ `NaN` sentinel.
    # If someone removes the remap, (b) fails loudly instead of quietly
    # leaking a NaN that `is.na()` would still catch.
    d <- make_regression_data(n = 20)
    model <- pprf(x = d$x, y = d$y, size = 3, seed = 0, threads = 1)
    n <- nrow(model$x)
    all_in_bag <- seq_len(n) - 1L
    for (i in seq_along(model$trees)) {
      model$trees[[i]]$sample_indices <- all_in_bag
    }
    if (!is.null(model$.cache) && exists("oob_predictions", envir = model$.cache, inherits = FALSE)) {
      rm("oob_predictions", envir = model$.cache)
    }
    preds <- oob_predictions(model)
    expect_true(all(is.na(preds)))
    expect_false(any(is.nan(preds)))
  })

  it("variable importance uses the continuous response", {
    # Regression VI (both permuted and weighted) needs the continuous
    # response as the OOB truth. Construct a dataset where x1 and x2 drive
    # y and x3 is pure noise; assert the informative columns outrank the
    # noise column for both measures.
    set.seed(7)
    n <- 100
    x <- matrix(runif(n * 3), ncol = 3)
    colnames(x) <- c("x1", "x2", "noise")
    y <- 2 * x[, 1] + 3 * x[, 2] + rnorm(n, sd = 0.1)

    model <- pprf(x = x, y = y, size = 20, seed = 0, threads = 1)
    expect_equal(model$mode, "regression")

    vi_perm     <- permuted_importance(model)
    vi_weighted <- weighted_importance(model)

    expect_length(vi_perm, 3)
    expect_length(vi_weighted, 3)
    expect_true(all(is.finite(vi_perm)))
    expect_true(all(is.finite(vi_weighted)))

    # Informative columns must outrank the noise column. (Not asserting
    # the ordering between x1 and x2 — coefficient-to-VI isn't monotonic
    # for small forests, and flakes across RNG seeds.)
    expect_gt(vi_perm[1], vi_perm[3])
    expect_gt(vi_perm[2], vi_perm[3])
    expect_gt(vi_weighted[1], vi_weighted[3])
    expect_gt(vi_weighted[2], vi_weighted[3])
  })

  it("VI is reproducible across runs", {
    set.seed(11)
    n <- 60
    x <- matrix(runif(n * 2), ncol = 2)
    y <- x[, 1] - x[, 2] + rnorm(n, sd = 0.05)

    m1 <- pprf(x = x, y = y, size = 10, seed = 0, threads = 1)
    m2 <- pprf(x = x, y = y, size = 10, seed = 0, threads = 1)

    expect_equal(permuted_importance(m1), permuted_importance(m2))
    expect_equal(weighted_importance(m1), weighted_importance(m2))
  })

  it("save/load round-trip preserves predictions", {
    d <- make_regression_data(n = 40)
    model <- pprf(x = d$x, y = d$y, size = 5, seed = 0, threads = 1)

    path <- tempfile(fileext = ".json")
    save_json(model, path)
    loaded <- load_json(path)

    expect_equal(loaded$mode, "regression")

    original_preds <- predict(model, d$x)
    loaded_preds <- predict(loaded, d$x)

    expect_equal(as.numeric(loaded_preds), as.numeric(original_preds))
  })

  it("rejects NA in features via matrix interface", {
    x <- matrix(c(1, NA, 3, 4, 1, 2, 3, 4), ncol = 2)
    y <- c(1.1, 1.2, 1.3, 1.4)
    expect_error(pprf(x = x, y = y, size = 3, threads = 1), "NA or NaN")
  })

  it("rejects NaN in features via matrix interface", {
    x <- matrix(c(1, NaN, 3, 4, 1, 2, 3, 4), ncol = 2)
    y <- c(1.1, 1.2, 1.3, 1.4)
    expect_error(pprf(x = x, y = y, size = 3, threads = 1), "NA or NaN")
  })

  it("trains with a constant feature column", {
    x <- cbind(rep(5, 6), c(1, 2, 3, 7, 8, 9))
    y <- c(0.1, 0.2, 0.3, 0.7, 0.8, 0.9)
    expect_no_error(pprf(x = x, y = y, size = 3, seed = 0, threads = 1))
  })

  it("trains with minimal dataset (n = 2)", {
    x <- matrix(c(1, 9, 0, 0), ncol = 2)
    y <- c(0.0, 1.0)
    expect_no_error(pprf(x = x, y = y, size = 3, seed = 0, threads = 1))
  })

  it("accepts valid p_vars (resolves to count)", {
    d <- make_regression_data(n = 40)
    model <- pprf(x = d$x, y = d$y, p_vars = 0.5, size = 2, seed = 0, threads = 1)
    # `make_regression_data` produces 2 features; 0.5 → max(1, 1) = 1.
    expect_equal(model$training_spec$vars$count, 1L)
  })

  it("trains with vars_all (no variable subsampling)", {
    d <- make_regression_data(n = 40)
    model <- pprf(x = d$x, y = d$y, size = 2, vars = vars_all(), seed = 0, threads = 1)
    expect_s3_class(model, "pprf_regression")
    expect_equal(model$training_spec$vars$name, "all")
  })
})

describe("pprf explicit mode parameter", {
  # See the matching block in test-pptr.R for the rationale — these
  # tests exercise the same `mode` plumbing on the forest entry point.

  it("mode = \"classification\" coerces integer 0/1 labels", {
    x <- matrix(rnorm(120), ncol = 2)
    y <- rep(c(0L, 1L), 30)
    model <- pprf(x = x, y = y, mode = "classification", size = 3, seed = 0, threads = 1)

    expect_equal(model$mode, "classification")
    expect_equal(length(model$groups), 2L)
    expect_s3_class(model, "pprf_classification")
  })

  it("mode = \"regression\" errors on a factor y", {
    expect_error(
      pprf(x = iris[, 1:4], y = iris$Species, mode = "regression", size = 3, seed = 0, threads = 1),
      "`mode = \"regression\"` requires a numeric response"
    )
  })

  it("mode = NULL preserves auto-detection (factor)", {
    model <- pprf(x = iris[, 1:4], y = iris$Species, mode = NULL, size = 3, seed = 0, threads = 1)
    expect_equal(model$mode, "classification")
  })

  it("mode = NULL preserves auto-detection (numeric)", {
    d <- make_regression_data()
    model <- pprf(x = d$x, y = d$y, mode = NULL, size = 3, seed = 0, threads = 1)
    expect_equal(model$mode, "regression")
  })

  it("rejects invalid mode strings", {
    expect_error(
      pprf(x = iris[, 1:4], y = iris$Species, mode = "ranking", size = 3, seed = 0, threads = 1),
      "`mode` must be either"
    )
  })
})

describe("pprf call capture and update()", {
  # `model$call` powers `stats::update.default`. We stick to checking the
  # contract (call is captured, update re-fits with the substituted arg)
  # rather than the exact deparsed form, so the test survives any future
  # cosmetic tweaks to print or to `match.call`'s output.
  it("stores the user's call on the fitted model", {
    model <- pprf(Species ~ ., data = iris, size = 3, lambda = 0.0, threads = 1)
    expect_false(is.null(model$call))
    expect_true(is.call(model$call))
    expect_equal(as.character(model$call)[[1L]], "pprf")
  })

  it("update() re-fits with the substituted argument", {
    set.seed(0)
    fit1 <- pprf(Species ~ ., data = iris, size = 3, lambda = 0.0, threads = 1)
    expect_equal(fit1$training_spec$pp$lambda, 0.0)

    fit2 <- update(fit1, lambda = 0.5)
    expect_s3_class(fit2, "pprf")
    expect_equal(fit2$training_spec$pp$lambda, 0.5)
    # Unchanged args carry over from the original call.
    expect_equal(length(fit2$trees), length(fit1$trees))
  })

  it("print() surfaces the call when present", {
    model <- pprf(Species ~ ., data = iris, size = 3, lambda = 0.0, threads = 1)
    out <- capture.output(print(model))
    expect_true(any(grepl("^\\s*Call:", out)))
    expect_true(any(grepl("pprf\\(", out)))
  })
})
