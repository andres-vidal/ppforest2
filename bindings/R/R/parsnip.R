# Suppress R CMD check NOTEs for rlang expressions used in parsnip registration
utils::globalVariables(c("object", "new_data"))

#' @keywords internal
.onLoad <- function(libname, pkgname) {
  if (requireNamespace("parsnip", quietly = TRUE)) {
    register_pp_rand_forest()
    register_pp_tree()
  }
}

#' Parsnip model specification for pprf.
#'
#' Creates a model specification for a Projection Pursuit random forest.
#' Use \code{set_engine("ppforest2")} to select the ppforest2 engine.
#'
#' @param mode A character string for the model type. Either \code{"classification"} or \code{"regression"}.
#' @param trees The number of trees in the forest (maps to \code{size}).
#' @param mtry The number of variables to consider at each split (maps to \code{n_vars}).
#' @param penalty The regularization parameter (maps to \code{lambda}).
#' @return A parsnip model specification.
#' @seealso \code{\link{pprf}} for the underlying training function, \code{\link{pp_tree}} for single trees
#' @examples
#' \dontrun{
#' library(parsnip)
#' spec <- pp_rand_forest(trees = 50, mtry = 2) %>% set_engine("ppforest2")
#' fit <- spec %>% fit(Species ~ ., data = iris)
#' predict(fit, iris)
#' predict(fit, iris, type = "prob")
#' }
#' @export
pp_rand_forest <- function(mode = "classification", trees = NULL, mtry = NULL, penalty = NULL) {
  if (!requireNamespace("parsnip", quietly = TRUE)) {
    stop("Package 'parsnip' is required for pp_rand_forest().", call. = FALSE)
  }

  args <- list(
    trees = rlang::enquo(trees),
    mtry = rlang::enquo(mtry),
    penalty = rlang::enquo(penalty)
  )

  parsnip::new_model_spec(
    "pp_rand_forest",
    args = args,
    eng_args = NULL,
    mode = mode,
    method = NULL,
    engine = NULL
  )
}

#' Parsnip model specification for pptr.
#'
#' Creates a model specification for a single Projection Pursuit decision tree.
#' Use \code{set_engine("ppforest2")} to select the ppforest2 engine.
#'
#' @param mode A character string for the model type. Either \code{"classification"} or \code{"regression"}.
#' @param penalty The regularization parameter (maps to \code{lambda}).
#' @return A parsnip model specification.
#' @seealso \code{\link{pptr}} for the underlying training function, \code{\link{pp_rand_forest}} for forests
#' @examples
#' \dontrun{
#' library(parsnip)
#' spec <- pp_tree(penalty = 0) %>% set_engine("ppforest2")
#' fit <- spec %>% fit(Species ~ ., data = iris)
#' predict(fit, iris)
#' }
#' @export
pp_tree <- function(mode = "classification", penalty = NULL) {
  if (!requireNamespace("parsnip", quietly = TRUE)) {
    stop("Package 'parsnip' is required for pp_tree().", call. = FALSE)
  }

  args <- list(
    penalty = rlang::enquo(penalty)
  )

  parsnip::new_model_spec(
    "pp_tree",
    args = args,
    eng_args = NULL,
    mode = mode,
    method = NULL,
    engine = NULL
  )
}

register_pp_rand_forest <- function() {
  try(parsnip::set_new_model("pp_rand_forest"), silent = TRUE)
  parsnip::set_model_mode("pp_rand_forest", "classification")
  parsnip::set_model_mode("pp_rand_forest", "regression")

  for (mode in c("classification", "regression")) {
    parsnip::set_model_engine("pp_rand_forest", mode = mode, eng = "ppforest2")

    parsnip::set_encoding(
      model = "pp_rand_forest",
      eng = "ppforest2",
      mode = mode,
      options = list(
        predictor_indicators = "none",
        compute_intercept = FALSE,
        remove_intercept = TRUE,
        allow_sparse_x = FALSE
      )
    )

    parsnip::set_fit(
      model = "pp_rand_forest",
      eng = "ppforest2",
      mode = mode,
      value = list(
        interface = "formula",
        protect = c("formula", "data"),
        func = c(pkg = "ppforest2", fun = "pprf"),
        # Forward the parsnip-declared mode to the engine function so it
        # round-trips through the wrapper instead of being re-derived from
        # `y`. parsnip's mode is the user's explicit intent; `pprf()`'s
        # `mode` arg then asserts on type compatibility (factor-y +
        # mode="regression" errors fast).
        defaults = list(mode = mode)
      )
    )
  }

  parsnip::set_model_arg(
    "pp_rand_forest", eng = "ppforest2",
    parsnip = "trees", original = "size",
    func = list(pkg = "dials", fun = "trees"),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    "pp_rand_forest", eng = "ppforest2",
    parsnip = "mtry", original = "n_vars",
    func = list(pkg = "dials", fun = "mtry"),
    has_submodel = FALSE
  )

  parsnip::set_model_arg(
    "pp_rand_forest", eng = "ppforest2",
    parsnip = "penalty", original = "lambda",
    func = list(pkg = "dials", fun = "penalty"),
    has_submodel = FALSE
  )

  parsnip::set_pred(
    model = "pp_rand_forest",
    eng = "ppforest2",
    mode = "classification",
    type = "class",
    value = list(
      pre = NULL,
      post = NULL,
      func = c(fun = "predict"),
      args = list(
        object = rlang::expr(object$fit),
        new_data = rlang::expr(new_data),
        type = "class"
      )
    )
  )

  parsnip::set_pred(
    model = "pp_rand_forest",
    eng = "ppforest2",
    mode = "classification",
    type = "prob",
    value = list(
      pre = NULL,
      post = NULL,
      func = c(fun = "predict"),
      args = list(
        object = rlang::expr(object$fit),
        new_data = rlang::expr(new_data),
        type = "prob"
      )
    )
  )

  parsnip::set_pred(
    model = "pp_rand_forest",
    eng = "ppforest2",
    mode = "regression",
    type = "numeric",
    value = list(
      pre = NULL,
      post = function(results, object) tibble::tibble(.pred = as.numeric(results)),
      func = c(fun = "predict"),
      args = list(
        object = rlang::expr(object$fit),
        new_data = rlang::expr(new_data),
        type = "response"
      )
    )
  )
}

register_pp_tree <- function() {
  try(parsnip::set_new_model("pp_tree"), silent = TRUE)
  parsnip::set_model_mode("pp_tree", "classification")
  parsnip::set_model_mode("pp_tree", "regression")

  for (mode in c("classification", "regression")) {
    parsnip::set_model_engine("pp_tree", mode = mode, eng = "ppforest2")

    parsnip::set_encoding(
      model = "pp_tree",
      eng = "ppforest2",
      mode = mode,
      options = list(
        predictor_indicators = "none",
        compute_intercept = FALSE,
        remove_intercept = TRUE,
        allow_sparse_x = FALSE
      )
    )

    parsnip::set_fit(
      model = "pp_tree",
      eng = "ppforest2",
      mode = mode,
      value = list(
        interface = "formula",
        protect = c("formula", "data"),
        func = c(pkg = "ppforest2", fun = "pptr"),
        # See the matching comment in `register_pp_rand_forest`.
        defaults = list(mode = mode)
      )
    )
  }

  parsnip::set_model_arg(
    "pp_tree", eng = "ppforest2",
    parsnip = "penalty", original = "lambda",
    func = list(pkg = "dials", fun = "penalty"),
    has_submodel = FALSE
  )

  parsnip::set_pred(
    model = "pp_tree",
    eng = "ppforest2",
    mode = "classification",
    type = "class",
    value = list(
      pre = NULL,
      post = NULL,
      func = c(fun = "predict"),
      args = list(
        object = rlang::expr(object$fit),
        new_data = rlang::expr(new_data),
        type = "class"
      )
    )
  )

  parsnip::set_pred(
    model = "pp_tree",
    eng = "ppforest2",
    mode = "classification",
    type = "prob",
    value = list(
      pre = NULL,
      post = NULL,
      func = c(fun = "predict"),
      args = list(
        object = rlang::expr(object$fit),
        new_data = rlang::expr(new_data),
        type = "prob"
      )
    )
  )

  parsnip::set_pred(
    model = "pp_tree",
    eng = "ppforest2",
    mode = "regression",
    type = "numeric",
    value = list(
      pre = NULL,
      post = function(results, object) tibble::tibble(.pred = as.numeric(results)),
      func = c(fun = "predict"),
      args = list(
        object = rlang::expr(object$fit),
        new_data = rlang::expr(new_data),
        type = "response"
      )
    )
  )
}
