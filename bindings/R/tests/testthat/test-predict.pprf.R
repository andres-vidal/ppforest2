Sys.setenv(R_TESTS = "")
Sys.setenv(OMP_THREAD_LIMIT = "1")
Sys.setenv(OMP_NUM_THREADS = "1")

library(testthat)
library(ppforest2)

describe("predict.pprf", {
  describe("on an object created with the formula interface", {
    it("returns a factor with the same length as the input matrix", {
      model <- pprf(Species ~ ., data = iris, threads = 1)
      predictions <- predict(model, iris)
      expect_equal(length(predictions), nrow(iris))
    })

    it("returns a factor with the same levels as the groups in the model", {
      model <- pprf(Species ~ ., data = iris, threads = 1)
      predictions <- predict(model, iris)
      expect_equal(levels(predictions), levels(iris$Species))
    })

    it("with new_data parameter returns the same result as positional", {
      model <- pprf(Species ~ ., data = iris, threads = 1)
      pred_positional <- predict(model, iris)
      pred_named <- predict(model, new_data = iris)
      expect_equal(pred_positional, pred_named)
    })
  })

  describe("on an object created with the matrix interface", {
    it("returns a factor with the same length as the input matrix", {
      x <- crabs[, 2:5]
      x$sex <- as.numeric(as.factor(crabs$sex))
      model <- pprf(x = x, y = crabs$Type, threads = 1)
      predictions <- predict(model, x)
      expect_equal(length(predictions), nrow(x))
    })

    it("returns a factor with the same levels as the groups in the model", {
      x <- crabs[, 2:5]
      x$sex <- as.numeric(as.factor(crabs$sex))
      model <- pprf(x = x, y = crabs$Type, threads = 1)
      predictions <- predict(model, x)
      expect_equal(levels(predictions), levels(crabs$Type))
    })
  })

  describe("with type = 'prob'", {
    it("returns a data frame with one column per group", {
      model <- pprf(Species ~ ., data = iris, threads = 1)
      probs <- predict(model, iris, type = "prob")
      expect_true(is.data.frame(probs))
      expect_equal(ncol(probs), length(levels(iris$Species)))
      expect_equal(colnames(probs), levels(iris$Species))
    })

    it("returns rows that sum to 1", {
      model <- pprf(Species ~ ., data = iris, threads = 1)
      probs <- predict(model, iris, type = "prob")
      row_sums <- rowSums(probs)
      expect_equal(row_sums, rep(1.0, nrow(iris)), tolerance = 1e-6)
    })

    it("returns values between 0 and 1", {
      model <- pprf(Species ~ ., data = iris, threads = 1)
      probs <- predict(model, iris, type = "prob")
      expect_true(all(probs >= 0))
      expect_true(all(probs <= 1))
    })

    it("returns the correct number of rows", {
      model <- pprf(Species ~ ., data = iris, threads = 1)
      probs <- predict(model, iris, type = "prob")
      expect_equal(nrow(probs), nrow(iris))
    })
  })
})
