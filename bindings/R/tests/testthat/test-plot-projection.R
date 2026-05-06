Sys.setenv(R_TESTS = "")
Sys.setenv(OMP_THREAD_LIMIT = "1")
Sys.setenv(OMP_NUM_THREADS = "1")

library(testthat)
library(ppforest2)

skip_if_not_installed("ggplot2")

describe("plot.pptr projection", {
  it("returns a ggplot object for type = 'projection'", {
    model <- pptr(Species ~ ., data = iris, seed = 0)
    p <- plot(model, type = "projection")
    expect_s3_class(p, "ggplot")
  })

  it("errors for out-of-range node index", {
    model <- pptr(Species ~ ., data = iris, seed = 0)
    expect_error(
      plot(model, type = "projection", node = 100L),
      "out of range"
    )
  })
})

describe("plot.pprf projection", {
  it("returns a ggplot object for type = 'projection'", {
    model <- pprf(Species ~ ., data = iris, size = 5, seed = 0, threads = 1)
    p <- plot(model, type = "projection", tree_index = 1)
    expect_s3_class(p, "ggplot")
  })
})

describe("plot.pptr projection snapshots", {
  skip_if_not_installed("vdiffr")
  skip_on_ci()  # vdiffr SVGs aren't byte-identical across platform LAPACK/BLAS

  model <- pptr(Species ~ ., data = iris, seed = 0)

  it("pptr-projection-root", {
    vdiffr::expect_doppelganger("pptr-projection-root", plot(model, type = "projection", node = 1L))
  })
})
