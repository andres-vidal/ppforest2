# Suppress R CMD check NOTEs for rlang expressions used in parsnip registration.
# `x` / `y` appear in `set_fit(... protect = c("x", "y"))` symbol references;
# `object` / `new_data` appear in `set_pred(... args = list(...))`.
utils::globalVariables(c("object", "new_data", "x", "y"))

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
#' @param mtry_prop The proportion of variables to consider at each split (maps to \code{p_vars}). An alternative to \code{mtry} that expresses the feature subsample as a fraction in (0, 1], tunable via the \code{dials} \code{mtry_prop()} parameter. Supply \code{mtry} or \code{mtry_prop}, not both.
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
pp_rand_forest <- function(mode = "classification", trees = NULL, mtry = NULL, mtry_prop = NULL, penalty = NULL) {
  if (!requireNamespace("parsnip", quietly = TRUE)) {
    stop("Package 'parsnip' is required for pp_rand_forest().", call. = FALSE)
  }

  args <- list(
    trees = rlang::enquo(trees),
    mtry = rlang::enquo(mtry),
    mtry_prop = rlang::enquo(mtry_prop),
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

#' Update a \code{pp_rand_forest} model specification.
#'
#' Implements parsnip's \code{update} protocol so that \code{tune::tune_grid()}
#' (and any other caller that finalises a spec via \code{update()}) can fill
#' in the tuned values for \code{trees}, \code{mtry}, and \code{penalty}.
#' Without this method, \code{update()} falls back to \code{stats::update.default}
#' and fails with "need an object with call component".
#'
#' @param object A \code{pp_rand_forest} model specification.
#' @param parameters A named list of parameters to update (alternative to passing them as args).
#' @param trees,mtry,mtry_prop,penalty New values for the corresponding parameters.
#' @param fresh If \code{TRUE}, drop any previously-set arg whose new value is not provided.
#' @param ... Engine-specific arguments to update.
#' @return An updated \code{pp_rand_forest} model specification.
#' @export
update.pp_rand_forest <- function(object, parameters = NULL, trees = NULL, mtry = NULL, mtry_prop = NULL, penalty = NULL, fresh = FALSE, ...) {
  args <- list(
    trees = rlang::enquo(trees),
    mtry = rlang::enquo(mtry),
    mtry_prop = rlang::enquo(mtry_prop),
    penalty = rlang::enquo(penalty)
  )
  update_pp_spec(object, parameters = parameters, args_enquo_list = args, fresh = fresh, cls = "pp_rand_forest", ...)
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

#' Update a \code{pp_tree} model specification.
#'
#' Companion to \code{\link{update.pp_rand_forest}} — exists for the same
#' reason (\code{tune::tune_grid()} relies on \code{update()} to finalise the
#' spec at each grid point).
#'
#' @param object A \code{pp_tree} model specification.
#' @param parameters A named list of parameters to update (alternative to passing them as args).
#' @param penalty A new value for the regularisation parameter.
#' @param fresh If \code{TRUE}, drop any previously-set arg whose new value is not provided.
#' @param ... Engine-specific arguments to update.
#' @return An updated \code{pp_tree} model specification.
#' @export
update.pp_tree <- function(object, parameters = NULL, penalty = NULL, fresh = FALSE, ...) {
  args <- list(
    penalty = rlang::enquo(penalty)
  )
  update_pp_spec(object, parameters = parameters, args_enquo_list = args, fresh = fresh, cls = "pp_tree", ...)
}

# Local re-implementation of parsnip's internal `update_spec()`. We avoid the
# triple-colon dependency to keep R CMD check clean. Behaviour mirrors
# `parsnip:::update_spec`: replace main args (and engine args from `...`)
# either by overlay (default) or completely (`fresh = TRUE`).
update_pp_spec <- function(object, parameters, args_enquo_list, fresh, cls, ...) {
  eng_dots <- rlang::enquos(...)
  if (fresh) {
    object$args <- args_enquo_list
    object$eng_args <- if (length(eng_dots)) eng_dots else NULL
  } else {
    is_null_arg <- vapply(args_enquo_list, function(q) {
      e <- rlang::quo_get_expr(q)
      is.null(e) || (is.symbol(e) && identical(e, quote(expr =)))
    }, logical(1))
    new_args <- args_enquo_list[!is_null_arg]
    if (length(new_args) > 0) {
      object$args[names(new_args)] <- new_args
    }
    if (length(eng_dots) > 0) {
      object$eng_args <- utils::modifyList(object$eng_args %||% list(), eng_dots)
    }
  }

  if (!is.null(parameters)) {
    if (!is.list(parameters) && !is.data.frame(parameters)) {
      stop("`parameters` must be a named list or one-row tibble.", call. = FALSE)
    }
    for (nm in names(parameters)) {
      val <- if (is.data.frame(parameters)) parameters[[nm]][[1]] else parameters[[nm]]
      object$args[[nm]] <- rlang::new_quosure(val, env = rlang::empty_env())
    }
  }

  parsnip::new_model_spec(
    cls,
    args = object$args,
    eng_args = object$eng_args,
    mode = object$mode,
    method = NULL,
    engine = object$engine
  )
}

register_pp_rand_forest <- function() {
  try(parsnip::set_new_model("pp_rand_forest"), silent = TRUE)
  parsnip::set_model_mode("pp_rand_forest", "classification")
  parsnip::set_model_mode("pp_rand_forest", "regression")

  for (mode in c("classification", "regression")) {
    parsnip::set_model_engine("pp_rand_forest", mode = mode, eng = "ppforest2")

    # Use the matrix interface (not formula) on purpose. With `interface =
    # "formula"`, parsnip's dispatch for x/y-input callers (workflows,
    # fit_resamples, tune_grid) is `xy_form()`, which synthesises a
    # `..y ~ x1 + x2 + ...` formula plus a data frame whose response column
    # is renamed to `..y`. `pprf()` then stores that synthetic formula on
    # the model, and prediction blows up with "object '..y' not found"
    # because the test data has no `..y` column. The matrix interface skips
    # that synthesis: parsnip's dispatch becomes `xy_xy(target = "matrix")`,
    # which calls the engine as `pprf(x = maybe_matrix(x), y = y, ...)`. We
    # also flip `predictor_indicators` from "none" to "traditional" so the
    # formula path (`form_xy()`) dummy-codes factor predictors into a numeric
    # matrix before handing them to `pprf()` — the old setup relied on
    # `pprf()` itself running `model.matrix()`, which doesn't happen on the
    # matrix path.
    parsnip::set_encoding(
      model = "pp_rand_forest",
      eng = "ppforest2",
      mode = mode,
      options = list(
        predictor_indicators = "traditional",
        compute_intercept = TRUE,
        remove_intercept = TRUE,
        allow_sparse_x = FALSE
      )
    )

    parsnip::set_fit(
      model = "pp_rand_forest",
      eng = "ppforest2",
      mode = mode,
      value = list(
        interface = "matrix",
        protect = c("x", "y"),
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
    parsnip = "mtry_prop", original = "p_vars",
    func = list(pkg = "dials", fun = "mtry_prop"),
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

    # Matrix interface — see the matching comment in
    # `register_pp_rand_forest` for the rationale (`..y` formula synthesis
    # in parsnip's `xy_form()` breaks prediction through workflows).
    parsnip::set_encoding(
      model = "pp_tree",
      eng = "ppforest2",
      mode = mode,
      options = list(
        predictor_indicators = "traditional",
        compute_intercept = TRUE,
        remove_intercept = TRUE,
        allow_sparse_x = FALSE
      )
    )

    parsnip::set_fit(
      model = "pp_tree",
      eng = "ppforest2",
      mode = mode,
      value = list(
        interface = "matrix",
        protect = c("x", "y"),
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
