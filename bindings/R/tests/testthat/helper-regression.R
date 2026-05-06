# Shared helper for regression tests.
# testthat auto-sources `helper-*.R` before any test file.

make_regression_data <- function(n = 30, seed = 0) {
  set.seed(seed)
  x <- matrix(runif(n * 2), ncol = 2)
  colnames(x) <- c("x1", "x2")
  y <- x[, 1] + x[, 2] + rnorm(n, sd = 0.1)
  list(x = x, y = y)
}
