Sys.setenv(R_TESTS = "")
Sys.setenv(OMP_THREAD_LIMIT = "1")
Sys.setenv(OMP_NUM_THREADS = "1")

library(testthat)
library(ppforest2)

skip_if_not_installed("ggplot2")

describe("plot.pptr boundaries", {
  it("returns a ggplot object for 2-feature boundary plot", {
    model <- pptr(
      x = iris[, c("Sepal.Length", "Sepal.Width")],
      y = iris$Species,
      seed = 0
    )
    p <- plot(model, type = "boundaries")
    expect_s3_class(p, "ggplot")
  })

  it("returns a gtable for 3-feature boundary plot (pairwise with stripped panels)", {
    model <- pptr(
      x = iris[, c("Sepal.Length", "Sepal.Width", "Petal.Length")],
      y = iris$Species,
      seed = 0
    )
    p <- plot(model, type = "boundaries")
    expect_s3_class(p, "gtable")
  })

  it("returns a gtable for pairwise boundary plot with 4 features", {
    model <- pptr(Species ~ ., data = iris, seed = 0)
    p <- plot(model, type = "boundaries")
    expect_s3_class(p, "gtable")
  })

  it("boundary plot uses C++ segment computation", {
    model <- pptr(
      x = iris[, c("Sepal.Length", "Sepal.Width")],
      y = iris$Species,
      seed = 0
    )
    segs <- ppforest2:::ppforest2_boundary_segments(
      model, c(0L, 1L), numeric(0),
      4.0, 8.0, 2.0, 4.5
    )
    expect_s3_class(segs, "data.frame")
    expect_true(nrow(segs) > 0)
    expect_true(all(c("x_start", "y_start", "x_end", "y_end", "depth") %in% names(segs)))
  })

  it("decision regions returns polygon data", {
    model <- pptr(
      x = iris[, c("Sepal.Length", "Sepal.Width")],
      y = iris$Species,
      seed = 0
    )
    regions <- ppforest2:::ppforest2_decision_regions(
      model, c(0L, 1L), numeric(0),
      4.0, 8.0, 2.0, 4.5
    )
    expect_true(is.list(regions))
    expect_true(length(regions) > 0)
    r <- regions[[1]]
    expect_true(all(c("x", "y", "group") %in% names(r)))
    expect_true(length(r$x) >= 3)
    expect_equal(length(r$x), length(r$y))
  })
})

describe("plot.pprf boundaries", {
  it("returns a ggplot object for 2-feature boundary plot", {
    model <- pprf(
      x = iris[, c("Sepal.Length", "Sepal.Width")],
      y = iris$Species,
      size = 5, seed = 0, threads = 1
    )
    p <- plot(model, type = "boundaries", tree_index = 1)
    expect_s3_class(p, "ggplot")
  })
})

describe("plot.pptr boundaries snapshots", {
  skip_if_not_installed("vdiffr")
  skip_on_ci()  # vdiffr SVGs aren't byte-identical across platform LAPACK/BLAS

  it("pptr-boundaries-2d", {
    model <- pptr(
      x = iris[, c("Sepal.Length", "Sepal.Width")],
      y = iris$Species,
      seed = 0
    )
    vdiffr::expect_doppelganger("pptr-boundaries-2d", plot(model, type = "boundaries"))
  })
})
