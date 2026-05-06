Sys.setenv(R_TESTS = "")
Sys.setenv(OMP_THREAD_LIMIT = "1")
Sys.setenv(OMP_NUM_THREADS = "1")

library(testthat)
library(ppforest2)

describe("formula.pptr", {
  it("on an object created with the formula interface, returns the formula used to create the model", {
    model <- pptr(Species ~ ., data = iris)
    expect_equal(
      formula(model),
      Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width - 1
    )
  })

  it("on an object created with the matrix interface, returns NULL", {
    model <- pptr(x = iris[, 1:4], y = iris[, 5])
    expect_equal(formula(model), NULL)
  })
})
