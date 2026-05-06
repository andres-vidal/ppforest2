Sys.setenv(R_TESTS = "")
Sys.setenv(OMP_THREAD_LIMIT = "1")
Sys.setenv(OMP_NUM_THREADS = "1")

library(testthat)
library(ppforest2)

describe("save_json / load_json round-trip", {
  describe("pptr (single tree)", {
    it("preserves predictions after round-trip", {
      model <- pptr(Species ~ ., data = iris, seed = 0)
      path <- tempfile(fileext = ".json")
      save_json(model, path)
      loaded <- load_json(path)

      x <- as.matrix(iris[, 1:4])
      original_preds <- predict(model, iris)
      loaded_preds <- predict(loaded, x)
      expect_equal(loaded_preds, original_preds)
    })

    it("preserves variable importance after round-trip", {
      model <- pptr(Species ~ ., data = iris, seed = 0)
      path <- tempfile(fileext = ".json")
      save_json(model, path)
      loaded <- load_json(path)

      expect_equal(unname(loaded$vi$scale), unname(model$vi$scale), tolerance = 1e-4)
      expect_equal(loaded$vi$projections, model$vi$projections, tolerance = 1e-4)
    })

    it("preserves group labels after round-trip", {
      model <- pptr(Species ~ ., data = iris, seed = 0)
      path <- tempfile(fileext = ".json")
      save_json(model, path)
      loaded <- load_json(path)

      expect_equal(loaded$groups, model$groups)
    })

    it("preserves training spec after round-trip", {
      model <- pptr(Species ~ ., data = iris, seed = 0, lambda = 0.5)
      path <- tempfile(fileext = ".json")
      save_json(model, path)
      loaded <- load_json(path)

      expect_equal(loaded$training_spec$pp$name, "pda")
      expect_equal(loaded$training_spec$pp$lambda, 0.5, tolerance = 1e-5)
      expect_equal(loaded$seed, 0)
    })

    it("sets formula, x, y to NULL on load", {
      model <- pptr(Species ~ ., data = iris, seed = 0)
      path <- tempfile(fileext = ".json")
      save_json(model, path)
      loaded <- load_json(path)

      expect_null(loaded$formula)
      expect_null(loaded$x)
      expect_null(loaded$y)
    })
  })

  describe("pprf regression", {
    # Regression has no group concept — `meta.groups` must be `null` on
    # disk and `character(0)` on the loaded model. The wrap also exposes
    # full `oob_metrics` (mse / mae / r_squared) instead of a flattened
    # scalar, mirroring the C++ Export shape.
    it("preserves predictions and metric structure after round-trip", {
      data(mtcars)
      model <- pprf(mpg ~ ., data = mtcars, size = 5, seed = 0, threads = 1)
      path <- tempfile(fileext = ".json")
      save_json(model, path)

      # Raw JSON: meta.groups is an empty array for regression (matches the
      # in-memory empty `Names`; the discriminator is `config.mode`).
      raw <- jsonlite::fromJSON(readLines(path, warn = FALSE), simplifyVector = FALSE)
      expect_equal(length(raw$meta$groups), 0)
      expect_equal(raw$config$mode, "regression")

      loaded <- load_json(path)
      expect_s3_class(loaded, "pprf_regression")
      expect_length(loaded$groups, 0)

      # Predictions match.
      x <- as.matrix(mtcars[, -1])
      expect_equal(predict(loaded, x), predict(model, mtcars), tolerance = 1e-5)

      # Full regression metrics block is exposed (not a single scalar).
      expect_true(is.list(loaded$oob_metrics))
      expect_named(loaded$oob_metrics, c("mse", "mae", "r_squared"), ignore.order = TRUE)
      expect_false(is.null(loaded$training_metrics))
      expect_named(loaded$training_metrics, c("mse", "mae", "r_squared"), ignore.order = TRUE)

      # `oob_error()` derives the scalar from the metrics block.
      err <- oob_error(loaded)
      expect_true(is.na(err) || isTRUE(all.equal(err, loaded$oob_metrics$mse, tolerance = 1e-5)))
    })
  })

  describe("pprf (random forest)", {
    it("preserves predictions after round-trip", {
      model <- pprf(Species ~ ., data = iris, size = 3, seed = 0)
      path <- tempfile(fileext = ".json")
      save_json(model, path)
      loaded <- load_json(path)

      x <- as.matrix(iris[, 1:4])
      original_preds <- predict(model, iris)
      loaded_preds <- predict(loaded, x)
      expect_equal(loaded_preds, original_preds)
    })

    it("preserves variable importance after round-trip", {
      model <- pprf(Species ~ ., data = iris, size = 3, seed = 0)
      path <- tempfile(fileext = ".json")
      save_json(model, path)
      loaded <- load_json(path)

      expect_equal(unname(loaded$vi$scale), unname(model$vi$scale), tolerance = 1e-4)
      expect_equal(loaded$vi$projections, model$vi$projections, tolerance = 1e-4)
      expect_equal(weighted_importance(loaded), weighted_importance(model), tolerance = 1e-4)
      expect_equal(permuted_importance(loaded), permuted_importance(model), tolerance = 1e-4)
    })

    it("preserves oob_error after round-trip", {
      model <- pprf(Species ~ ., data = iris, size = 3, seed = 0)
      path <- tempfile(fileext = ".json")
      save_json(model, path)
      loaded <- load_json(path)

      expect_equal(oob_error(loaded), oob_error(model), tolerance = 1e-5)
    })

    it("exposes full classification metrics block after round-trip", {
      # Classification mirrors the regression test above: full
      # `training_metrics` / `oob_metrics` blocks (not flattened scalars),
      # each carrying the confusion matrix and error rate. The CM's labels
      # are group names (strings), and its dimensions match meta.groups
      # in the well-balanced case.
      model <- pprf(Species ~ ., data = iris, size = 3, seed = 0)
      path <- tempfile(fileext = ".json")
      save_json(model, path)
      loaded <- load_json(path)

      expect_true(is.list(loaded$training_metrics))
      expect_named(loaded$training_metrics, c("confusion_matrix", "error_rate"),
        ignore.order = TRUE)
      cm <- loaded$training_metrics$confusion_matrix
      expect_named(cm, c("matrix", "labels", "group_errors"), ignore.order = TRUE)
      expect_equal(dim(cm$matrix), c(length(loaded$groups), length(loaded$groups)))

      expect_true(is.list(loaded$oob_metrics))
      expect_named(loaded$oob_metrics, c("confusion_matrix", "error_rate"),
        ignore.order = TRUE)
    })

    it("CM labels equal meta.groups (no prediction-only labels in normal use)", {
      # Asserts the C++ contract that, in normal training/evaluation,
      # `predictions ⊆ groups` and `actual ⊆ groups`, so the CM's
      # `labels` set equals `meta.groups`. If a future change introduces
      # prediction-only labels (a class the model never predicted, or one
      # it predicted that never appeared in `y`), this assertion fires
      # and the R-side CM helper in `test-reproducibility.R` needs an
      # update to handle the asymmetric case.
      model <- pprf(Species ~ ., data = iris, size = 3, seed = 0)
      path <- tempfile(fileext = ".json")
      save_json(model, path)
      raw <- jsonlite::fromJSON(readLines(path, warn = FALSE), simplifyVector = FALSE)

      cm_labels <- vapply(raw$training_metrics$confusion_matrix$labels,
        as.character, character(1))
      expect_equal(cm_labels, unlist(raw$meta$groups))
    })

    it("preserves group labels on individual trees", {
      model <- pprf(Species ~ ., data = iris, size = 3, seed = 0)
      path <- tempfile(fileext = ".json")
      save_json(model, path)
      loaded <- load_json(path)

      for (i in seq_along(loaded$trees)) {
        expect_s3_class(loaded$trees[[i]], "pptr")
        expect_equal(loaded$trees[[i]]$groups, model$groups)
      }
    })
  })
})

describe("save_json includes data metadata", {
  it("includes observations, features, and feature_names for tree", {
    model <- pptr(Species ~ ., data = iris, seed = 0)
    path <- tempfile(fileext = ".json")
    save_json(model, path)

    j <- jsonlite::fromJSON(readLines(path, warn = FALSE))
    expect_equal(j$meta$observations, 150)
    expect_equal(j$meta$features, 4)
    expect_equal(j$meta$feature_names, c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"))
  })

  it("includes observations, features, and feature_names for forest", {
    model <- pprf(Species ~ ., data = iris, size = 3, seed = 0)
    path <- tempfile(fileext = ".json")
    save_json(model, path)

    j <- jsonlite::fromJSON(readLines(path, warn = FALSE))
    expect_equal(j$meta$observations, 150)
    expect_equal(j$meta$features, 4)
    expect_equal(j$meta$feature_names, c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"))
  })

})

describe("save_json with include_metrics = FALSE", {
  it("saves model without VI", {
    model <- pptr(Species ~ ., data = iris, seed = 0)
    path <- tempfile(fileext = ".json")
    save_json(model, path, include_metrics = FALSE)
    loaded <- load_json(path)

    expect_null(loaded$vi)

    x <- as.matrix(iris[, 1:4])
    original_preds <- predict(model, iris)
    loaded_preds <- predict(loaded, x)
    expect_equal(loaded_preds, original_preds)
  })
})

describe("load_json from golden files", {
  golden_path <- function(...) {
    system.file("golden", ..., package = "ppforest2")
  }

  it("loads a golden tree and produces correct predictions", {
    path <- golden_path("iris", "tree-pda-s0.json")
    skip_if(path == "", "Golden file not bundled")

    loaded <- load_json(path)
    expect_s3_class(loaded, "pptr")
    expect_equal(loaded$groups, c("setosa", "versicolor", "virginica"))

    x <- as.matrix(iris[, 1:4])
    preds <- predict(loaded, x)
    expect_length(preds, 150)
  })

  it("loads a golden forest", {
    path <- golden_path("iris", "forest-pda-n5-s0.json")
    skip_if(path == "", "Golden file not bundled")

    loaded <- load_json(path)
    expect_s3_class(loaded, "pprf")
    expect_equal(length(loaded$trees), 5)
  })
})

describe("R save_json meta matches golden file", {
  golden_path <- function(...) {
    system.file("golden", ..., package = "ppforest2")
  }

  golden_meta <- function(dataset, slug) {
    path <- golden_path(dataset, paste0(slug, ".json"))
    if (path == "") return(NULL)
    j <- jsonlite::fromJSON(readLines(path, warn = FALSE))
    j$meta
  }

  it("tree meta matches golden file", {
    expected <- golden_meta("iris", "tree-pda-s0")
    skip_if(is.null(expected), "Golden file not bundled")

    model <- pptr(Species ~ ., data = iris, seed = 0)
    path <- tempfile(fileext = ".json")
    save_json(model, path)
    j <- jsonlite::fromJSON(readLines(path, warn = FALSE))

    expect_equal(j$meta, expected)
  })

  it("forest meta matches golden file", {
    expected <- golden_meta("iris", "forest-pda-n5-s0")
    skip_if(is.null(expected), "Golden file not bundled")

    model <- pprf(Species ~ ., data = iris, size = 5, seed = 0)
    path <- tempfile(fileext = ".json")
    save_json(model, path)
    j <- jsonlite::fromJSON(readLines(path, warn = FALSE))

    expect_equal(j$meta, expected)
  })
})

describe("save_json / load_json with non-default strategies", {
  it("preserves tree training spec (pp, vars, cutpoint) after round-trip", {
    model <- pptr(Species ~ ., data = iris, pp = pp_pda(0.5), seed = 0)
    path <- tempfile(fileext = ".json")
    save_json(model, path)
    loaded <- load_json(path)

    expect_equal(loaded$training_spec$pp$name, "pda")
    expect_equal(loaded$training_spec$pp$lambda, 0.5, tolerance = 1e-5)
    expect_equal(loaded$training_spec$vars$name, "all")
    expect_equal(loaded$training_spec$cutpoint$name, "mean_of_means")
  })

  it("preserves forest training spec with vars_uniform after round-trip", {
    model <- pprf(Species ~ ., data = iris, size = 3, pp = pp_pda(0.3), vars = vars_uniform(n_vars = 2), seed = 0)
    path <- tempfile(fileext = ".json")
    save_json(model, path)
    loaded <- load_json(path)

    expect_equal(loaded$training_spec$pp$name, "pda")
    expect_equal(loaded$training_spec$pp$lambda, 0.3, tolerance = 1e-5)
    expect_equal(loaded$training_spec$vars$name, "uniform")
    expect_equal(loaded$training_spec$vars$count, 2)
    expect_equal(loaded$training_spec$cutpoint$name, "mean_of_means")
  })

  it("preserves forest training spec with vars_all after round-trip", {
    model <- pprf(Species ~ ., data = iris, size = 3, vars = vars_all(), seed = 0)
    path <- tempfile(fileext = ".json")
    save_json(model, path)
    loaded <- load_json(path)

    expect_equal(loaded$training_spec$vars$name, "all")
  })

  it("does not include display_name in saved JSON", {
    model <- pptr(Species ~ ., data = iris, pp = pp_pda(0.5), seed = 0)
    path <- tempfile(fileext = ".json")
    save_json(model, path)

    j <- jsonlite::fromJSON(readLines(path, warn = FALSE))
    expect_null(j$config$pp$display_name)
    expect_null(j$config$vars$display_name)
    expect_null(j$config$cutpoint$display_name)
  })

  it("predictions are correct after loading model with non-default strategies", {
    model <- pprf(Species ~ ., data = iris, size = 3, pp = pp_pda(0.3), vars = vars_uniform(n_vars = 2), seed = 0)
    path <- tempfile(fileext = ".json")
    save_json(model, path)
    loaded <- load_json(path)

    original_preds <- predict(model, iris)
    loaded_preds <- predict(loaded, model$x)
    expect_equal(loaded_preds, original_preds)
  })
})

describe("load_json error handling", {
  it("errors on invalid JSON", {
    path <- tempfile(fileext = ".json")
    writeLines("not valid json", path)
    expect_error(load_json(path))
  })
})
