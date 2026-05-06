Sys.setenv(R_TESTS = "")
Sys.setenv(OMP_THREAD_LIMIT = "1")
Sys.setenv(OMP_NUM_THREADS = "1")

library(testthat)
library(ppforest2)

describe("pptr formula interface", {
  it("returns a object with s3 class pptr", {
    model <- pptr(Species ~ ., data = iris)
    expect_s3_class(model, "pptr")
  })

  it("preserves the observations (x) as a matrix in the returned model", {
    model <- pptr(Species ~ ., data = iris)
    expected <- model.matrix(
      Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width - 1,
      iris,
      response = TRUE
    )
    expect_equal(model$x, expected)
  })

  it("preserves the labels (y) as a matrix in the returned model", {
    model <- pptr(Species ~ ., data = iris)
    expected <- as.matrix(as.numeric(as.factor(iris$Species)))
    expect_equal(model$y, expected)
  })

  it("preserves the groups in the returned model", {
    model <- pptr(Species ~ ., data = iris)
    expected <- levels(iris$Species)
    expect_equal(model$groups, expected)
  })

  it("preserves the formula in the returned model", {
    model <- pptr(Species ~ ., data = iris)
    expect_equal(
      model$formula,
      Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width - 1
    )
  })
})

describe("pptr matrix interface", {
  it("returns a object with s3 class pptr", {
    model <- pptr(x = iris[, 1:4], y = iris[, 5])
    expect_s3_class(model, "pptr")
  })

  it("preserves the observations (x) as a matrix in the returned model", {
    model <- pptr(x = iris[, 1:4], y = iris[, 5])
    expected <- as.matrix(iris[, 1:4])
    expect_equal(model$x, expected)
  })

  it("preserves the labels (y) as a matrix in the returned model", {
    model <- pptr(x = iris[, 1:4], y = iris[, 5])
    expected <- as.matrix(as.numeric(as.factor(iris$Species)))
    expect_equal(model$y, expected)
  })

  it("preserves the groups in the returned model", {
    model <- pptr(x = iris[, 1:4], y = iris[, 5])
    expected <- levels(iris$Species)
    expect_equal(model$groups, expected)
  })

  it("does not have a formula", {
    model <- pptr(x = iris[, 1:4], y = iris[, 5])
    expect_equal(model$formula, NULL)
  })
})

describe("pptr reproducibility", {
  it("produces identical predictions with set.seed", {
    set.seed(0)
    model1 <- pptr(Species ~ ., data = iris)
    set.seed(0)
    model2 <- pptr(Species ~ ., data = iris)
    expect_equal(predict(model1, iris), predict(model2, iris))
  })

  it("produces identical predictions with explicit seed", {
    model1 <- pptr(Species ~ ., data = iris, seed = 123L)
    model2 <- pptr(Species ~ ., data = iris, seed = 123L)
    expect_equal(predict(model1, iris), predict(model2, iris))
  })

  it("stores the explicit seed in the model", {
    model <- pptr(Species ~ ., data = iris, seed = 0)
    expect_equal(model$seed, 0)
  })

  it("stores the generated seed when seed is NULL", {
    set.seed(99)
    model <- pptr(Species ~ ., data = iris)
    expect_true(is.numeric(model$seed))
    expect_true(model$seed > 0)
  })
})

describe("pptr input validation", {
  it("rejects invalid lambda", {
    expect_error(pptr(x = iris[, 1:4], y = iris[, 5], lambda = -1), "between 0 and 1")
    expect_error(pptr(x = iris[, 1:4], y = iris[, 5], lambda = 2), "between 0 and 1")
    expect_error(pptr(x = iris[, 1:4], y = iris[, 5], lambda = "a"), "between 0 and 1")
  })

  it("rejects non-integer seed", {
    expect_error(pptr(x = iris[, 1:4], y = iris[, 5], seed = 1.5), "integer")
  })

  it("rejects single-group y", {
    expect_error(pptr(x = iris[, 1:4], y = rep("a", 150)), "at least 2 groups")
  })

  it("rejects dimension mismatch", {
    expect_error(pptr(x = iris[1:10, 1:4], y = iris[, 5]), "same number of observations")
  })

  # Non-finite y rejection — used to cross the C++ boundary and trip an
  # out-of-range cast. Rejecting here fails fast with a clear message.
  it("rejects NA in factor y", {
    x <- as.matrix(iris[, 1:4])
    y <- iris$Species
    y[10] <- NA
    expect_error(pptr(x = x, y = y), "`y` must not contain NA")
  })
})

describe("pptr training spec", {
  it("preserves the lambda parameter in the returned model", {
    model <- pptr(Species ~ ., data = iris, lambda = 0.5)
    expect_equal(model$training_spec$pp$lambda, 0.5)
  })

  it("the lambda parameter is 0 by default", {
    model <- pptr(Species ~ ., data = iris)
    expect_equal(model$training_spec$pp$lambda, 0)
  })

  it("the pp strategy is pda", {
    model <- pptr(Species ~ ., data = iris)
    expect_equal(model$training_spec$pp$name, "pda")
  })

  it("the vars strategy is all", {
    model <- pptr(Species ~ ., data = iris)
    expect_equal(model$training_spec$vars$name, "all")
  })

  it("the stop strategy is pure_node", {
    model <- pptr(Species ~ ., data = iris)
    expect_equal(model$training_spec$stop$name, "pure_node")
  })

  it("the binarize strategy is largest_gap", {
    model <- pptr(Species ~ ., data = iris)
    expect_equal(model$training_spec$binarize$name, "largest_gap")
  })

  it("the grouping strategy is by_label", {
    model <- pptr(Species ~ ., data = iris)
    expect_equal(model$training_spec$grouping$name, "by_label")
  })

  it("the leaf strategy is majority_vote", {
    model <- pptr(Species ~ ., data = iris)
    expect_equal(model$training_spec$leaf$name, "majority_vote")
  })
})

describe("pptr regression training spec", {
  # Pins the mode-aware defaults selected by
  # `TrainingSpec::Builder::apply_defaults()` for regression. The C++
  # side is the single source of truth; this block guards against
  # silent drift if the default selection ever changes.
  it("the pp strategy is pda", {
    d <- make_regression_data()
    model <- pptr(x = d$x, y = d$y, seed = 0)
    expect_equal(model$training_spec$pp$name, "pda")
  })

  it("the vars strategy is all", {
    d <- make_regression_data()
    model <- pptr(x = d$x, y = d$y, seed = 0)
    expect_equal(model$training_spec$vars$name, "all")
  })

  it("the cutpoint strategy is mean_of_means", {
    d <- make_regression_data()
    model <- pptr(x = d$x, y = d$y, seed = 0)
    expect_equal(model$training_spec$cutpoint$name, "mean_of_means")
  })

  it("the stop strategy is any(min_size(5), min_variance(0.01))", {
    d <- make_regression_data()
    model <- pptr(x = d$x, y = d$y, seed = 0)
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
    model <- pptr(x = d$x, y = d$y, seed = 0)
    expect_equal(model$training_spec$binarize$name, "disabled")
  })

  it("the grouping strategy is by_cutpoint", {
    d <- make_regression_data()
    model <- pptr(x = d$x, y = d$y, seed = 0)
    expect_equal(model$training_spec$grouping$name, "by_cutpoint")
  })

  it("the leaf strategy is mean_response", {
    d <- make_regression_data()
    model <- pptr(x = d$x, y = d$y, seed = 0)
    expect_equal(model$training_spec$leaf$name, "mean_response")
  })
})

describe("pptr with strategy objects", {
  it("trains with pp = pp_pda()", {
    model <- pptr(Species ~ ., data = iris, pp = pp_pda(0.5), seed = 0)
    expect_s3_class(model, "pptr")
    expect_equal(model$training_spec$pp$lambda, 0.5)
  })

  it("trains with pp = pp_pda(0) (LDA)", {
    model <- pptr(Species ~ ., data = iris, pp = pp_pda(0), seed = 0)
    expect_s3_class(model, "pptr")
    expect_equal(model$training_spec$pp$lambda, 0)
  })

  it("produces identical export as lambda shortcut", {
    model_shortcut <- pptr(Species ~ ., data = iris, lambda = 0.5, seed = 0)
    model_strategy <- pptr(Species ~ ., data = iris, pp = pp_pda(0.5), seed = 0)

    path_shortcut <- tempfile(fileext = ".json")
    path_strategy <- tempfile(fileext = ".json")
    save_json(model_shortcut, path_shortcut)
    save_json(model_strategy, path_strategy)

    j_shortcut <- jsonlite::fromJSON(readLines(path_shortcut, warn = FALSE))
    j_strategy <- jsonlite::fromJSON(readLines(path_strategy, warn = FALSE))

    # Compare full export (model, config, meta, metrics)
    j_shortcut$training_duration_ms <- NULL
    j_strategy$training_duration_ms <- NULL
    expect_equal(j_shortcut, j_strategy)
  })

  it("trains with explicit stop, binarize, grouping, leaf", {
    model <- pptr(Species ~ ., data = iris, stop = stop_pure_node(), binarize = binarize_largest_gap(), grouping = grouping_by_label(), leaf = leaf_majority_vote(), seed = 0)
    expect_s3_class(model, "pptr")
    expect_equal(model$training_spec$stop$name, "pure_node")
    expect_equal(model$training_spec$binarize$name, "largest_gap")
    expect_equal(model$training_spec$grouping$name, "by_label")
    expect_equal(model$training_spec$leaf$name, "majority_vote")
  })

  it("rejects non-stop_strategy objects", {
    expect_error(
      pptr(Species ~ ., data = iris, stop = list(name = "pure_node")),
      "stop_strategy")
  })

  it("rejects non-binarize_strategy objects", {
    expect_error(
      pptr(Species ~ ., data = iris, binarize = list(name = "largest_gap")),
      "binarize_strategy")
  })

  it("rejects non-grouping_strategy objects", {
    expect_error(
      pptr(Species ~ ., data = iris, grouping = list(name = "by_label")),
      "grouping_strategy")
  })

  it("rejects non-leaf_strategy objects", {
    expect_error(
      pptr(Species ~ ., data = iris, leaf = list(name = "majority_vote")),
      "leaf_strategy")
  })

  it("default tree and fully explicit defaults produce identical export", {
    model_default  <- pptr(Species ~ ., data = iris, seed = 0)
    model_explicit <- pptr(Species ~ ., data = iris, pp = pp_pda(0), cutpoint = cutpoint_mean_of_means(), stop = stop_pure_node(), binarize = binarize_largest_gap(), grouping = grouping_by_label(), leaf = leaf_majority_vote(), seed = 0)

    path_default  <- tempfile(fileext = ".json")
    path_explicit <- tempfile(fileext = ".json")
    save_json(model_default, path_default)
    save_json(model_explicit, path_explicit)

    j_default  <- jsonlite::fromJSON(readLines(path_default, warn = FALSE))
    j_explicit <- jsonlite::fromJSON(readLines(path_explicit, warn = FALSE))

    j_default$training_duration_ms  <- NULL
    j_explicit$training_duration_ms <- NULL
    expect_equal(j_default, j_explicit)
  })

  it("errors when mixing pp and lambda", {
    expect_error(
      pptr(Species ~ ., data = iris, pp = pp_pda(0.5), lambda = 0.3),
      "Cannot use `pp` together with `lambda`")
  })

  it("rejects non-pp_strategy objects", {
    expect_error(
      pptr(Species ~ ., data = iris, pp = list(name = "pda", lambda = 0.5)),
      "pp_strategy")
  })

  it("rejects non-cutpoint_strategy objects", {
    expect_error(
      pptr(Species ~ ., data = iris, cutpoint = list(name = "mean_of_means")),
      "cutpoint_strategy")
  })
})

describe("pptr edge cases", {
  it("rejects NA in features via matrix interface", {
    x <- matrix(c(1, NA, 3, 4, 1, 2, 3, 4), ncol = 2)
    y <- c("a", "a", "b", "b")
    expect_error(pptr(x = x, y = y), "NA or NaN")
  })

  it("rejects NaN in features via matrix interface", {
    x <- matrix(c(1, NaN, 3, 4, 1, 2, 3, 4), ncol = 2)
    y <- c("a", "a", "b", "b")
    expect_error(pptr(x = x, y = y), "NA or NaN")
  })

  it("trains with a constant feature column", {
    df <- data.frame(x1 = rep(5, 6), x2 = c(1, 2, 3, 7, 8, 9), y = c("a", "a", "a", "b", "b", "b"))
    expect_no_error(pptr(y ~ ., data = df, seed = 0))
  })

  it("trains with single observation per group", {
    df <- data.frame(x1 = c(1, 0), x2 = c(0, 1), y = c("a", "b"))
    model <- pptr(y ~ ., data = df, seed = 0)
    expect_s3_class(model, "pptr")
    preds <- predict(model, df)
    expect_length(preds, 2)
  })

  it("trains with minimal dataset (n=2)", {
    df <- data.frame(x1 = c(1, 9), y = c("a", "b"))
    model <- pptr(y ~ ., data = df, seed = 0)
    expect_s3_class(model, "pptr")
    preds <- predict(model, df)
    expect_length(preds, 2)
  })

  it("trains with extreme class imbalance", {
    df <- data.frame(
      x1 = c(0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 90, 91),
      x2 = c(0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 90, 91),
      y = c(rep("a", 18), rep("b", 2))
    )
    model <- pptr(y ~ ., data = df, seed = 0)
    expect_s3_class(model, "pptr")
    preds <- predict(model, df)
    expect_length(preds, 20)
  })
})

describe("pptr regression", {
  it("end-to-end on mtcars (formula + predict + summary)", {
    # Real-dataset round-trip counterpart to the simulated-data tests
    # below, mirroring the analog in `test-pprf.R`. mtcars is small
    # (32 x 11) and ships with base R so this doesn't grow the package
    # footprint, while still exercising the full single-tree pipeline
    # on real covariances: formula interface, regression auto-detection
    # from numeric `y`, training, prediction, and summary rendering.
    data(mtcars)
    model <- pptr(mpg ~ ., data = mtcars, seed = 0)

    expect_equal(model$mode, "regression")
    # `groups` is an empty character vector for regression (no class levels).
    expect_length(model$groups, 0)
    expect_s3_class(model, "pptr_regression")
    expect_s3_class(model, "pptr")
    expect_s3_class(model, "ppmodel")

    preds <- predict(model, newdata = mtcars)
    expect_true(is.numeric(preds))
    expect_length(preds, nrow(mtcars))
    expect_true(all(is.finite(preds)))

    # summary() must render the regression-shaped header + training
    # metrics block (MSE / MAE / R²) and must not render a confusion
    # matrix. The "Regression Tree" header distinguishes from the
    # classification "Decision Tree" header.
    out <- capture.output(summary(model))
    expect_true(any(grepl("Regression Tree", out)))
    expect_true(any(grepl("Training Metrics", out)))
    expect_true(any(grepl("MSE|R\\^2|R²|MAE", out)))
    expect_false(any(grepl("Confusion Matrix", out)))
  })

  it("auto-detects regression from numeric y", {
    d <- make_regression_data()
    model <- pptr(x = d$x, y = d$y, seed = 0)

    expect_equal(model$mode, "regression")
    expect_length(model$groups, 0)
    # Unified `y`: continuous response stored on `model$y` for both modes.
    expect_equal(length(model$y), 30)
  })

  it("still dispatches to classification when y is a factor", {
    model <- pptr(x = iris[, 1:4], y = iris$Species, seed = 0)

    expect_equal(model$mode, "classification")
    expect_equal(length(model$groups), 3)
  })

  it("predict returns a numeric vector", {
    d <- make_regression_data()
    model <- pptr(x = d$x, y = d$y, seed = 0)
    preds <- predict(model, d$x)

    expect_type(preds, "double")
    expect_length(preds, 30)
    expect_false(is.factor(preds))
  })

  it("predict rejects type='class' and type='prob'", {
    d <- make_regression_data()
    model <- pptr(x = d$x, y = d$y, seed = 0)

    expect_error(predict(model, d$x, type = "class"), "regression")
    expect_error(predict(model, d$x, type = "prob"), "regression")
  })

  it("supports the formula interface", {
    d <- make_regression_data()
    df <- data.frame(x1 = d$x[, 1], x2 = d$x[, 2], y = d$y)
    model <- pptr(y ~ ., data = df, seed = 0)

    expect_equal(model$mode, "regression")
    preds <- predict(model, df)
    expect_type(preds, "double")
  })

  it("save/load round-trip preserves predictions", {
    d <- make_regression_data()
    model <- pptr(x = d$x, y = d$y, seed = 0)

    path <- tempfile(fileext = ".json")
    save_json(model, path)
    loaded <- load_json(path)

    expect_equal(loaded$mode, "regression")

    original_preds <- predict(model, d$x)
    loaded_preds <- predict(loaded, d$x)

    expect_equal(as.numeric(loaded_preds), as.numeric(original_preds))
  })

  it("works with custom strategies", {
    d <- make_regression_data(n = 50)
    model <- pptr(
      x = d$x, y = d$y, seed = 0,
      stop = stop_min_size(10),
      leaf = leaf_mean_response(),
      grouping = grouping_by_cutpoint()
    )

    expect_equal(model$mode, "regression")
    preds <- predict(model, d$x)
    expect_length(preds, 50)
  })

  it("is reproducible with the same seed", {
    d <- make_regression_data()

    m1 <- pptr(x = d$x, y = d$y, seed = 42)
    m2 <- pptr(x = d$x, y = d$y, seed = 42)

    expect_equal(as.numeric(predict(m1, d$x)), as.numeric(predict(m2, d$x)))
  })

  # Non-finite y rejection — these values used to cross the C++ boundary
  # and either trip an out-of-range cast or violate strict-weak-order in
  # std::stable_sort. Rejecting in `validate_data` fails fast.
  it("rejects NA in y", {
    d <- make_regression_data()
    d$y[3] <- NA_real_
    expect_error(pptr(x = d$x, y = d$y), "`y` must not contain NA")
  })

  it("rejects NaN in y", {
    d <- make_regression_data()
    d$y[5] <- NaN
    expect_error(pptr(x = d$x, y = d$y), "`y` must not contain NA")
  })

  it("rejects Inf in y", {
    d <- make_regression_data()
    d$y[7] <- Inf
    expect_error(pptr(x = d$x, y = d$y), "finite values for regression")
  })

  it("rejects NA in features via matrix interface", {
    x <- matrix(c(1, NA, 3, 4, 1, 2, 3, 4), ncol = 2)
    y <- c(1.1, 1.2, 1.3, 1.4)
    expect_error(pptr(x = x, y = y), "NA or NaN")
  })

  it("rejects NaN in features via matrix interface", {
    x <- matrix(c(1, NaN, 3, 4, 1, 2, 3, 4), ncol = 2)
    y <- c(1.1, 1.2, 1.3, 1.4)
    expect_error(pptr(x = x, y = y), "NA or NaN")
  })

  it("trains with a constant feature column", {
    # First column is constant; the second carries the signal. The
    # variable selector / cutpoint must handle the degenerate column
    # without aborting training.
    x <- cbind(rep(5, 6), c(1, 2, 3, 7, 8, 9))
    y <- c(0.1, 0.2, 0.3, 0.7, 0.8, 0.9)
    expect_no_error(pptr(x = x, y = y, seed = 0))
  })

  it("trains with minimal dataset (n = 2)", {
    # ByCutpoint::compute_init handles n = 2 by splitting [0, 0] / [1, 1].
    # The R side accepts any n >= 1 (matches the C++ minimum).
    x <- matrix(c(1, 9, 0, 0), ncol = 2)
    y <- c(0.0, 1.0)
    expect_no_error(pptr(x = x, y = y, seed = 0))
  })
})

describe("pptr explicit mode parameter", {
  # The `mode` arg lets callers override the auto-detection from `y`'s
  # type (factor/character -> classification, numeric -> regression).

  it("mode = \"classification\" coerces integer 0/1 labels", {
    # Common binary-labels case: a numeric vector that would otherwise
    # auto-detect as regression. Explicit mode pins classification, and
    # the labels are coerced via `factor()`.
    x <- matrix(rnorm(40), ncol = 2)
    y <- rep(c(0L, 1L), 10)
    model <- pptr(x = x, y = y, mode = "classification", seed = 0)

    expect_equal(model$mode, "classification")
    expect_equal(length(model$groups), 2L)
    expect_s3_class(model, "pptr_classification")
  })

  it("mode = \"regression\" errors on a factor y", {
    # Type-mismatch fail-fast: factor `y` cannot be coerced to a
    # meaningful continuous response.
    expect_error(
      pptr(x = iris[, 1:4], y = iris$Species, mode = "regression", seed = 0),
      "`mode = \"regression\"` requires a numeric response"
    )
  })

  it("mode = NULL preserves auto-detection (factor)", {
    model <- pptr(x = iris[, 1:4], y = iris$Species, mode = NULL, seed = 0)
    expect_equal(model$mode, "classification")
  })

  it("mode = NULL preserves auto-detection (numeric)", {
    d <- make_regression_data()
    model <- pptr(x = d$x, y = d$y, mode = NULL, seed = 0)
    expect_equal(model$mode, "regression")
  })

  it("rejects invalid mode strings", {
    expect_error(
      pptr(x = iris[, 1:4], y = iris$Species, mode = "ranking", seed = 0),
      "`mode` must be either"
    )
  })
})
