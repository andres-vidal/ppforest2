Sys.setenv(R_TESTS = "")
Sys.setenv(OMP_THREAD_LIMIT = "1")
Sys.setenv(OMP_NUM_THREADS = "1")

library(testthat)
library(ppforest2)

skip_if_not_installed("parsnip")

library(parsnip)

describe("parsnip integration", {
  describe("pp_rand_forest", {
    it("can be created as a parsnip model specification", {
      spec <- pp_rand_forest() |>
        set_engine("ppforest2") |>
        set_mode("classification")
      expect_s3_class(spec, "pp_rand_forest")
    })

    it("can be fit using a parsnip workflow", {
      spec <- pp_rand_forest() |>
        set_engine("ppforest2") |>
        set_mode("classification")
      fit <- spec |> fit(Species ~ ., data = iris)
      expect_s3_class(fit, "model_fit")
      expect_s3_class(fit$fit, "pprf")
    })

    it("can predict groups via parsnip", {
      spec <- pp_rand_forest() |>
        set_engine("ppforest2") |>
        set_mode("classification")
      fit <- spec |> fit(Species ~ ., data = iris)
      preds <- predict(fit, iris)
      expect_s3_class(preds, "tbl_df")
      expect_equal(nrow(preds), nrow(iris))
      expect_true(".pred_class" %in% colnames(preds))
    })

    it("can predict probabilities via parsnip", {
      spec <- pp_rand_forest() |>
        set_engine("ppforest2") |>
        set_mode("classification")
      fit <- spec |> fit(Species ~ ., data = iris)
      preds <- predict(fit, iris, type = "prob")
      expect_s3_class(preds, "tbl_df")
      expect_equal(nrow(preds), nrow(iris))
      prob_cols <- grep("^\\.pred_", colnames(preds), value = TRUE)
      expect_equal(length(prob_cols), length(levels(iris$Species)))
    })
  })

  describe("pp_tree", {
    it("can be created as a parsnip model specification", {
      spec <- pp_tree() |>
        set_engine("ppforest2") |>
        set_mode("classification")
      expect_s3_class(spec, "pp_tree")
    })

    it("can be fit using a parsnip workflow", {
      spec <- pp_tree() |>
        set_engine("ppforest2") |>
        set_mode("classification")
      fit <- spec |> fit(Species ~ ., data = iris)
      expect_s3_class(fit, "model_fit")
      expect_s3_class(fit$fit, "pptr")
    })

    it("can predict groups via parsnip", {
      spec <- pp_tree() |>
        set_engine("ppforest2") |>
        set_mode("classification")
      fit <- spec |> fit(Species ~ ., data = iris)
      preds <- predict(fit, iris)
      expect_s3_class(preds, "tbl_df")
      expect_equal(nrow(preds), nrow(iris))
      expect_true(".pred_class" %in% colnames(preds))
    })

    it("can predict probabilities via parsnip", {
      spec <- pp_tree() |>
        set_engine("ppforest2") |>
        set_mode("classification")
      fit <- spec |> fit(Species ~ ., data = iris)
      preds <- predict(fit, iris, type = "prob")
      expect_s3_class(preds, "tbl_df")
      expect_equal(nrow(preds), nrow(iris))
    })
  })

  describe("regression mode", {
    make_reg_df <- function() {
      set.seed(0)
      df <- data.frame(x1 = runif(50), x2 = runif(50))
      df$y <- df$x1 + df$x2 + rnorm(50, sd = 0.1)
      df
    }

    it("pp_tree can fit and predict regression", {
      spec <- pp_tree(mode = "regression") |> set_engine("ppforest2")
      df <- make_reg_df()
      fit <- spec |> fit(y ~ ., data = df)
      preds <- predict(fit, df)
      expect_s3_class(preds, "tbl_df")
      expect_equal(nrow(preds), nrow(df))
      expect_true(".pred" %in% colnames(preds))
      expect_true(is.numeric(preds$.pred))
    })

    it("pp_rand_forest can fit and predict regression", {
      spec <- pp_rand_forest(mode = "regression", trees = 10) |> set_engine("ppforest2")
      df <- make_reg_df()
      fit <- spec |> fit(y ~ ., data = df)
      preds <- predict(fit, df)
      expect_s3_class(preds, "tbl_df")
      expect_equal(nrow(preds), nrow(df))
      expect_true(".pred" %in% colnames(preds))
      expect_true(is.numeric(preds$.pred))
    })
  })

  describe("seed reproducibility through parsnip", {
    # parsnip's `fit()` does not wrap the engine call in a private RNG
    # context, so `pprf`'s internal `sample.int()` draws from the caller's
    # R RNG state. Lock in the three ways a user can control reproducibility:
    #   (1) outer `set.seed()`
    #   (2) `set_engine("ppforest2", seed = X)`  [engine-arg forwarding]
    #   (3) `withr::with_seed()`                 [tune's idiom]
    # If any of these regresses, reproducibility-by-seed under parsnip
    # silently breaks.
    it("outer set.seed flows through parsnip fit", {
      spec <- pp_rand_forest(trees = 5) |> set_engine("ppforest2")

      set.seed(42)
      fit1 <- spec |> fit(Species ~ ., data = iris)
      set.seed(42)
      fit2 <- spec |> fit(Species ~ ., data = iris)

      expect_identical(fit1$fit$seed, fit2$fit$seed)
      expect_equal(predict(fit1, iris), predict(fit2, iris))
    })

    it("set_engine(seed = X) is forwarded to pprf and reproducible", {
      spec <- pp_rand_forest(trees = 5) |> set_engine("ppforest2", seed = 777L)

      fit1 <- spec |> fit(Species ~ ., data = iris)
      fit2 <- spec |> fit(Species ~ ., data = iris)

      expect_identical(fit1$fit$seed, 777L)
      expect_identical(fit2$fit$seed, 777L)
      expect_equal(predict(fit1, iris), predict(fit2, iris))
    })

    it("engine-arg seed overrides outer set.seed", {
      spec <- pp_rand_forest(trees = 5) |> set_engine("ppforest2", seed = 777L)

      set.seed(1)
      fit_a <- spec |> fit(Species ~ ., data = iris)
      set.seed(999)
      fit_b <- spec |> fit(Species ~ ., data = iris)

      # Engine arg wins: both fits use seed 777 regardless of outer state.
      expect_identical(fit_a$fit$seed, 777L)
      expect_identical(fit_b$fit$seed, 777L)
      expect_equal(predict(fit_a, iris), predict(fit_b, iris))
    })

    it("withr::with_seed controls the engine RNG (tune-package idiom)", {
      skip_if_not_installed("withr")
      spec <- pp_rand_forest(trees = 5) |> set_engine("ppforest2")

      fit1 <- withr::with_seed(100, spec |> fit(Species ~ ., data = iris))
      fit2 <- withr::with_seed(100, spec |> fit(Species ~ ., data = iris))

      expect_identical(fit1$fit$seed, fit2$fit$seed)
      expect_equal(predict(fit1, iris), predict(fit2, iris))
    })
  })
})
