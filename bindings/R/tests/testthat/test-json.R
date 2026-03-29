Sys.setenv(R_TESTS = "")
Sys.setenv(OMP_THREAD_LIMIT = "1")
Sys.setenv(OMP_NUM_THREADS = "1")

library(testthat)
library(ppforest2)

describe("save_json / load_json round-trip", {
  describe("pptr (single tree)", {
    it("preserves predictions after round-trip", {
      model <- pptr(Type ~ ., data = iris, seed = 42)
      path <- tempfile(fileext = ".json")
      save_json(model, path)
      loaded <- load_json(path)

      x <- as.matrix(iris[, 1:4])
      original_preds <- predict(model, iris)
      loaded_preds <- predict(loaded, x)
      expect_equal(loaded_preds, original_preds)
    })

    it("preserves variable importance after round-trip", {
      model <- pptr(Type ~ ., data = iris, seed = 42)
      path <- tempfile(fileext = ".json")
      save_json(model, path)
      loaded <- load_json(path)

      expect_equal(unname(loaded$vi$scale), unname(model$vi$scale), tolerance = 1e-4)
      expect_equal(loaded$vi$projections, model$vi$projections, tolerance = 1e-4)
    })

    it("preserves group labels after round-trip", {
      model <- pptr(Type ~ ., data = iris, seed = 42)
      path <- tempfile(fileext = ".json")
      save_json(model, path)
      loaded <- load_json(path)

      expect_equal(loaded$groups, model$groups)
    })

    it("preserves training spec after round-trip", {
      model <- pptr(Type ~ ., data = iris, seed = 42, lambda = 0.5)
      path <- tempfile(fileext = ".json")
      save_json(model, path)
      loaded <- load_json(path)

      expect_equal(loaded$training_spec$strategy, "pda")
      expect_equal(loaded$training_spec$lambda, 0.5, tolerance = 1e-5)
      expect_equal(loaded$seed, 42L)
    })

    it("sets formula, x, y to NULL on load", {
      model <- pptr(Type ~ ., data = iris, seed = 42)
      path <- tempfile(fileext = ".json")
      save_json(model, path)
      loaded <- load_json(path)

      expect_null(loaded$formula)
      expect_null(loaded$x)
      expect_null(loaded$y)
    })
  })

  describe("pprf (random forest)", {
    it("preserves predictions after round-trip", {
      model <- pprf(Type ~ ., data = iris, size = 3, seed = 42)
      path <- tempfile(fileext = ".json")
      save_json(model, path)
      loaded <- load_json(path)

      x <- as.matrix(iris[, 1:4])
      original_preds <- predict(model, iris)
      loaded_preds <- predict(loaded, x)
      expect_equal(loaded_preds, original_preds)
    })

    it("preserves variable importance after round-trip", {
      model <- pprf(Type ~ ., data = iris, size = 3, seed = 42)
      path <- tempfile(fileext = ".json")
      save_json(model, path)
      loaded <- load_json(path)

      expect_equal(unname(loaded$vi$scale), unname(model$vi$scale), tolerance = 1e-4)
      expect_equal(loaded$vi$projections, model$vi$projections, tolerance = 1e-4)
      expect_equal(loaded$vi$weighted, model$vi$weighted, tolerance = 1e-4)
      expect_equal(loaded$vi$permuted, model$vi$permuted, tolerance = 1e-4)
    })

    it("preserves oob_error after round-trip", {
      model <- pprf(Type ~ ., data = iris, size = 3, seed = 42)
      path <- tempfile(fileext = ".json")
      save_json(model, path)
      loaded <- load_json(path)

      expect_equal(loaded$oob_error, model$oob_error, tolerance = 1e-5)
    })

    it("preserves group labels on individual trees", {
      model <- pprf(Type ~ ., data = iris, size = 3, seed = 42)
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
    model <- pptr(Type ~ ., data = iris, seed = 42)
    path <- tempfile(fileext = ".json")
    save_json(model, path)

    j <- jsonlite::fromJSON(readLines(path, warn = FALSE))
    expect_equal(j$meta$observations, 150)
    expect_equal(j$meta$features, 4)
    expect_equal(j$meta$feature_names, c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"))
  })

  it("includes observations, features, and feature_names for forest", {
    model <- pprf(Type ~ ., data = iris, size = 3, seed = 42)
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
    model <- pptr(Type ~ ., data = iris, seed = 42)
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
    path <- golden_path("iris", "tree-pda-s42.json")
    skip_if(path == "", "Golden file not bundled")

    loaded <- load_json(path)
    expect_s3_class(loaded, "pptr")
    expect_equal(loaded$groups, c("setosa", "versicolor", "virginica"))

    x <- as.matrix(iris[, 1:4])
    preds <- predict(loaded, x)
    expect_length(preds, 150)
  })

  it("loads a golden forest", {
    path <- golden_path("iris", "forest-pda-t5-s42.json")
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
    expected <- golden_meta("iris", "tree-pda-s42")
    skip_if(is.null(expected), "Golden file not bundled")

    model <- pptr(Type ~ ., data = iris, seed = 42)
    path <- tempfile(fileext = ".json")
    save_json(model, path)
    j <- jsonlite::fromJSON(readLines(path, warn = FALSE))

    expect_equal(j$meta, expected)
  })

  it("forest meta matches golden file", {
    expected <- golden_meta("iris", "forest-pda-t5-s42")
    skip_if(is.null(expected), "Golden file not bundled")

    model <- pprf(Type ~ ., data = iris, size = 5, seed = 42)
    path <- tempfile(fileext = ".json")
    save_json(model, path)
    j <- jsonlite::fromJSON(readLines(path, warn = FALSE))

    expect_equal(j$meta, expected)
  })
})

describe("load_json error handling", {
  it("errors on invalid JSON", {
    path <- tempfile(fileext = ".json")
    writeLines("not valid json", path)
    expect_error(load_json(path))
  })
})
