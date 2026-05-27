Sys.setenv(R_TESTS = "")
Sys.setenv(OMP_THREAD_LIMIT = "1")
Sys.setenv(OMP_NUM_THREADS = "1")

library(testthat)
library(ppforest2)

# Coverage for the standard R fit-class generics: fitted(), residuals(),
# nobs(). These are what downstream tools (broom::glance, step(), AIC(),
# users' ggplot diagnostic plots) call without thinking. Also locks in
# the `predict(fit, matrix)` path, which used to error whenever the model
# was fit via the formula interface and a caller wanted to predict on a
# raw numeric matrix (e.g. `model$x`).
describe("standard R accessors", {
  describe("nobs()", {
    it("returns the training-row count for pprf", {
      m <- pprf(Species ~ ., data = iris, size = 3, threads = 1)
      expect_equal(nobs(m), nrow(iris))
    })

    it("returns the training-row count for pptr", {
      m <- pptr(Species ~ ., data = iris)
      expect_equal(nobs(m), nrow(iris))
    })

    it("returns NA when the model has no training data attached", {
      m <- pprf(Species ~ ., data = iris, size = 3, threads = 1)
      m$x <- NULL
      expect_true(is.na(nobs(m)))
    })
  })

  describe("fitted()", {
    it("returns a factor matching the training response for classification", {
      m <- pprf(Species ~ ., data = iris, size = 5, threads = 1)
      fv <- fitted(m)
      expect_s3_class(fv, "factor")
      expect_equal(length(fv), nrow(iris))
      # In-sample accuracy on iris should be near-perfect for a 5-tree forest
      # (this is a sanity check, not a quality claim — see oob_predictions()
      # for an honest estimate).
      expect_gt(mean(fv == iris$Species), 0.9)
    })

    it("returns a numeric vector for regression", {
      set.seed(0)
      df <- data.frame(x1 = runif(40), x2 = runif(40))
      df$y <- df$x1 + 0.5 * df$x2 + rnorm(40, sd = 0.1)
      m <- pprf(y ~ ., data = df, size = 5, threads = 1)
      fv <- fitted(m)
      expect_true(is.numeric(fv))
      expect_equal(length(fv), nrow(df))
    })

    it("works on pptr too", {
      m <- pptr(Species ~ ., data = iris)
      expect_s3_class(fitted(m), "factor")
    })
  })

  describe("residuals()", {
    it("returns y - fitted for regression models", {
      set.seed(0)
      df <- data.frame(x1 = runif(40), x2 = runif(40))
      df$y <- df$x1 + rnorm(40, sd = 0.1)
      m <- pprf(y ~ ., data = df, size = 5, threads = 1)
      r <- residuals(m)
      expect_true(is.numeric(r))
      expect_equal(length(r), nrow(df))
      # `pprf` sorts rows by y in regression mode, so `model$y` is NOT in the
      # original input order. The residual contract is "training y minus
      # in-sample prediction" — both are taken from the (sorted) training
      # state, not from the caller's pre-sort data frame.
      expect_equal(r, as.numeric(m$y) - as.numeric(fitted(m)), tolerance = 1e-10)
    })

    it("errors with a clear message on classification models", {
      m <- pprf(Species ~ ., data = iris, size = 3, threads = 1)
      expect_error(residuals(m), "only defined for regression")
    })
  })

  describe("predict() accepts a numeric matrix", {
    # Regression: previously `predict(fit, fit$x)` errored with "data must
    # be a data.frame" whenever the model was fit via the formula interface.
    # Now matrix input bypasses model.matrix.
    it("accepts model$x (matrix) directly for formula-fitted pprf", {
      m <- pprf(Species ~ ., data = iris, size = 3, threads = 1)
      p <- predict(m, m$x)
      expect_s3_class(p, "factor")
      expect_equal(length(p), nrow(iris))
    })

    it("accepts a matrix for formula-fitted pptr", {
      m <- pptr(Species ~ ., data = iris)
      p <- predict(m, m$x)
      expect_s3_class(p, "factor")
      expect_equal(length(p), nrow(iris))
    })

    it("still accepts a data.frame (regression path)", {
      m <- pprf(Species ~ ., data = iris, size = 3, threads = 1)
      expect_s3_class(predict(m, iris), "factor")
    })
  })
})
