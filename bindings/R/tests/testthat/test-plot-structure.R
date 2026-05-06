Sys.setenv(R_TESTS = "")
Sys.setenv(OMP_THREAD_LIMIT = "1")
Sys.setenv(OMP_NUM_THREADS = "1")

library(testthat)
library(ppforest2)

skip_if_not_installed("ggplot2")

describe("plot.pptr structure", {
  it("returns a ggplot object for type = 'structure'", {
    model <- pptr(Species ~ ., data = iris, seed = 0)
    p <- plot(model, type = "structure")
    expect_s3_class(p, "ggplot")
  })

  it("tree plot uses C++ node data with histograms", {
    model <- pptr(Species ~ ., data = iris, seed = 0)
    nodes <- ppforest2:::ppforest2_tree_node_data(model, model$x, model$y)
    expect_true(length(nodes) > 0)
    expect_false(nodes[[1]]$is_leaf)
    expect_true(length(nodes[[1]]$projected) > 0)
    expect_true(length(nodes[[1]]$groups) > 0)
  })

  it("default plot renders mosaic without error", {
    model <- pptr(Species ~ ., data = iris, seed = 0)
    expect_no_error(plot(model))
  })
})

describe("plot.pprf structure", {
  it("returns a ggplot object for type = 'structure'", {
    model <- pprf(Species ~ ., data = iris, size = 5, seed = 0, threads = 1)
    p <- plot(model, type = "structure", tree_index = 1)
    expect_s3_class(p, "ggplot")
  })
})

describe("plot.pptr structure snapshots", {
  skip_if_not_installed("vdiffr")
  skip_on_ci()  # vdiffr SVGs aren't byte-identical across platform LAPACK/BLAS

  model <- pptr(Species ~ ., data = iris, seed = 0)

  it("pptr-structure", {
    vdiffr::expect_doppelganger("pptr-structure", plot(model, type = "structure"))
  })
})

describe("plot.pprf structure snapshots", {
  skip_if_not_installed("vdiffr")
  skip_on_ci()  # vdiffr SVGs aren't byte-identical across platform LAPACK/BLAS

  model <- pprf(Species ~ ., data = iris, size = 5, seed = 0, threads = 1)

  it("pprf-structure", {
    vdiffr::expect_doppelganger("pprf-structure", plot(model, type = "structure", tree_index = 1))
  })
})

describe("plot.pptr regression structure snapshot", {
  # Regression trees render their leaves as mean-response values (numeric
  # labels) rather than majority-vote class names. The snapshot fences
  # that difference: if a future change accidentally re-routes regression
  # leaves through the classification renderer (or vice versa), the
  # rendered text labels will diverge from the saved image.
  skip_if_not_installed("vdiffr")
  skip_on_ci()  # vdiffr SVGs aren't byte-identical across platform LAPACK/BLAS

  data(mtcars)
  model <- pptr(mpg ~ ., data = mtcars, seed = 0)

  it("pptr-regression-structure", {
    vdiffr::expect_doppelganger("pptr-regression-structure", plot(model, type = "structure"))
  })
})

describe("format_projector", {
  it("formats top-k terms by magnitude", {
    fmt <- ppforest2:::format_projector(
      c(0.1, -0.7, 0.3, 0.05),
      c("a", "b", "c", "d"),
      max_terms = 2L
    )
    expect_true(grepl("b", fmt))
    expect_true(grepl("c", fmt))
    expect_true(grepl("\u2026", fmt))
  })

  it("shows all terms when fewer than max_terms", {
    fmt <- ppforest2:::format_projector(
      c(0.5, -0.3),
      c("x1", "x2"),
      max_terms = 3L
    )
    expect_true(grepl("x1", fmt))
    expect_true(grepl("x2", fmt))
    expect_false(grepl("\u2026", fmt))
  })

  it("returns '0' for a zero projector", {
    fmt <- ppforest2:::format_projector(
      c(0, 0, 0),
      c("a", "b", "c"),
      max_terms = 3L
    )
    expect_equal(fmt, "0")
  })
})
