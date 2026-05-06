Sys.setenv(R_TESTS = "")
Sys.setenv(OMP_THREAD_LIMIT = "1")
Sys.setenv(OMP_NUM_THREADS = "1")

library(testthat)
library(ppforest2)

skip_if_not_installed("ggplot2")

describe("plot.pptr importance", {
  it("returns a ggplot object for type = 'importance'", {
    model <- pptr(Species ~ ., data = iris, seed = 0)
    p <- plot(model, type = "importance")
    expect_s3_class(p, "ggplot")
  })

  it("importance plot orders variables by value", {
    model <- pptr(Species ~ ., data = iris, seed = 0)
    p <- plot(model, type = "importance")
    pdata <- ggplot2::ggplot_build(p)$layout$panel_params[[1]]$y$get_labels()
    vi <- model$vi$projections
    vnames <- colnames(model$x)
    expected_order <- vnames[order(vi)]
    expect_equal(pdata, expected_order)
  })
})

describe("plot.pprf importance", {
  it("default plot renders importance grid without error", {
    model <- pprf(Species ~ ., data = iris, size = 5, seed = 0, threads = 1)
    expect_no_error(plot(model))
  })

  it("renders importance grid for type = 'importance' without metric", {
    model <- pprf(Species ~ ., data = iris, size = 5, seed = 0, threads = 1)
    expect_no_error(plot(model, type = "importance"))
  })

  it("returns a ggplot for a single importance metric", {
    model <- pprf(Species ~ ., data = iris, size = 5, seed = 0, threads = 1)
    p <- plot(model, metric = "projections")
    expect_s3_class(p, "ggplot")
    p <- plot(model, metric = "weighted")
    expect_s3_class(p, "ggplot")
    p <- plot(model, metric = "permuted")
    expect_s3_class(p, "ggplot")
  })

  it("importance plot orders variables by selected metric", {
    model <- pprf(Species ~ ., data = iris, size = 5, seed = 0, threads = 1)
    p <- plot(model, metric = "permuted")
    pdata <- ggplot2::ggplot_build(p)$layout$panel_params[[1]]$y$get_labels()
    vi <- permuted_importance(model)
    vnames <- colnames(model$x)
    expected_order <- vnames[order(vi)]
    expect_equal(pdata, expected_order)
  })
})

describe("plot.pptr importance snapshots", {
  skip_if_not_installed("vdiffr")
  skip_on_ci()  # vdiffr SVGs aren't byte-identical across platform LAPACK/BLAS

  model <- pptr(Species ~ ., data = iris, seed = 0)

  it("pptr-importance", {
    vdiffr::expect_doppelganger("pptr-importance", plot(model, type = "importance"))
  })
})

describe("plot.pprf importance snapshots", {
  skip_if_not_installed("vdiffr")
  skip_on_ci()  # vdiffr SVGs aren't byte-identical across platform LAPACK/BLAS

  model <- pprf(Species ~ ., data = iris, size = 5, seed = 0, threads = 1)

  it("pprf-importance-projections", {
    vdiffr::expect_doppelganger("pprf-importance-projections", plot(model, metric = "projections"))
  })

  it("pprf-importance-permuted", {
    vdiffr::expect_doppelganger("pprf-importance-permuted", plot(model, metric = "permuted"))
  })
})

describe("plot.pprf regression importance snapshot", {
  # Regression-side VI share the three measures with classification
  # (projections / permuted / weighted), but the underlying numbers come
  # from MSE-increase (regression) rather than accuracy-drop
  # (classification). The snapshot fences the end-to-end render for a
  # regression model so future refactors of the VI-computation path
  # (unit scale, sign handling, label formatting) can't silently alter
  # the plotted output.
  skip_if_not_installed("vdiffr")
  skip_on_ci()  # vdiffr SVGs aren't byte-identical across platform LAPACK/BLAS

  data(mtcars)
  model <- pprf(mpg ~ ., data = mtcars, size = 5, seed = 0, threads = 1)

  it("pprf-regression-importance-projections", {
    vdiffr::expect_doppelganger(
      "pprf-regression-importance-projections",
      plot(model, metric = "projections")
    )
  })
})
