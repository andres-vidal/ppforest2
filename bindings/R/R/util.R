# Null-coalescing helper. Package-private; used from json.R, stop-strategy.R,
# and anywhere else that wants rlang's `%||%` without depending on rlang.
`%||%` <- function(a, b) if (is.null(a)) b else a


# Resolve strategy objects vs shortcut params into strategy objects.
#
# When shortcut params are used (lambda, n_vars, p_vars), builds the
# corresponding strategy objects. When explicit strategy objects are
# provided, validates and forwards them. Errors if both APIs are mixed.
#
# When n_features is provided, p_vars is resolved to n_vars and the
# upper bound (n_vars <= n_features) is validated for uniform variable selection.
#
# @param pp A pp_strategy object or NULL.
# @param lambda Shortcut for PDA lambda (caller must pass missing() result).
# @param lambda_missing TRUE if lambda was not explicitly passed by the user.
# @param vars A vars_strategy object or NULL.
# @param n_vars Shortcut for number of variables.
# @param n_vars_missing TRUE if n_vars was not explicitly passed by the user.
# @param p_vars Shortcut for proportion of variables.
# @param p_vars_missing TRUE if p_vars was not explicitly passed by the user.
# @param cutpoint A cutpoint_strategy object or NULL.
# @param stop A stop_strategy object or NULL.
# @param binarize A binarize_strategy object or NULL.
# @param grouping A grouping_strategy object or NULL.
# @param leaf A leaf_strategy object or NULL.
# @param default_vars The default variable selection strategy when none is specified.
# @param n_features Number of features, used to resolve p_vars and validate
#   n_vars upper bound. NULL if not yet known.
# @return A list with resolved strategy objects.
resolve_strategies <- function(
  pp,
  lambda,
  lambda_missing,
  vars = NULL,
  n_vars = NULL,
  n_vars_missing = TRUE,
  p_vars = NULL,
  p_vars_missing = TRUE,
  cutpoint = NULL,
  stop = NULL,
  binarize = NULL,
  grouping = NULL,
  leaf = NULL,
  default_vars = vars_all(),
  n_features = NULL) {
  if (!is.null(pp) && !lambda_missing)
    stop("Cannot use `pp` together with `lambda`. Use one API or the other.")

  if (!is.null(vars) && (!n_vars_missing || !p_vars_missing))
    stop("Cannot use `vars` together with `n_vars`/`p_vars`. Use one API or the other.")

  if (!is.null(pp) && !inherits(pp, "pp_strategy"))
    stop("`pp` must be a pp_strategy object (e.g., pp_pda()).")

  if (!is.null(vars) && !inherits(vars, "vars_strategy"))
    stop("`vars` must be a vars_strategy object (e.g., vars_uniform() or vars_all()).")

  if (!is.null(cutpoint) && !inherits(cutpoint, "cutpoint_strategy"))
    stop("`cutpoint` must be a cutpoint_strategy object (e.g., cutpoint_mean_of_means()).")

  if (!is.null(stop) && !inherits(stop, "stop_strategy"))
    stop("`stop` must be a stop_strategy object (e.g., stop_pure_node()).")

  if (!is.null(binarize) && !inherits(binarize, "binarize_strategy"))
    stop("`binarize` must be a binarize_strategy object (e.g., binarize_largest_gap()).")

  if (!is.null(grouping) && !inherits(grouping, "grouping_strategy"))
    stop("`grouping` must be a grouping_strategy object (e.g., grouping_by_label()).")

  if (!is.null(leaf) && !inherits(leaf, "leaf_strategy"))
    stop("`leaf` must be a leaf_strategy object (e.g., leaf_majority_vote()).")

  # PP strategy
  if (is.null(pp))
    pp <- pp_pda(lambda)

  # Variable selection strategy
  if (is.null(vars)) {
    if (!is.null(n_vars) || !is.null(p_vars)) {
      vars <- vars_uniform(n_vars = n_vars, p_vars = p_vars)
    } else {
      vars <- default_vars
    }
  }

  # Resolve p_vars to count now that we know the number of features
  if (!is.null(n_features) && vars$name == "uniform") {
    if (!is.null(vars$p_vars)) {
      vars$count <- max(1L, as.integer(round(vars$p_vars * n_features)))
      vars$p_vars <- NULL
    }
    if (is.null(vars$count)) {
      vars$count <- n_features
    }
    if (vars$count > n_features) {
      stop("`n_vars` must be an integer between 1 and the number of features (", n_features, ").")
    }
  }

  # Strategy defaults (cutpoint, stop, binarize, grouping, leaf) are
  # resolved by `TrainingSpec::Builder::apply_defaults()` on the C++ side
  # — the single source of truth for mode-aware defaults. Anything left
  # `NULL` here is dropped from the training-spec list passed to C++,
  # and the Builder fills it in at `make()` time. R-side resolution
  # remains responsible only for parsing user-facing shortcuts (lambda,
  # n_vars, p_vars) and validating user-supplied strategy objects.

  list(
    pp = pp,
    vars = vars,
    cutpoint = cutpoint,
    stop = stop,
    binarize = binarize,
    grouping = grouping,
    leaf = leaf)
}

print_training_spec <- function(spec) {
  section_labels <- c(
    pp = "pp method", vars = "vars method", cutpoint = "cutpoint method",
    stop = "stop rule", binarize = "binarize method", grouping = "grouping method",
    leaf = "leaf method")
  for (section in c("pp", "vars", "cutpoint", "stop", "binarize", "grouping", "leaf")) {
    s <- spec[[section]]
    if (is.null(s)) next
    display <- if (!is.null(s$display_name)) s$display_name else s$name
    params <- s[!names(s) %in% c("name", "display_name")]
    if (length(params) == 0) {
      cat(section_labels[[section]], ": ", display, "\n", sep = "")
    } else {
      param_str <- paste(names(params), params, sep = "=", collapse = ", ")
      cat(section_labels[[section]], ": ", display, " (", param_str, ")\n", sep = "")
    }
  }
  cat("\n")
}

print_confusion_matrix <- function(raw_preds, model) {
  preds <- factor(model$groups[raw_preds], levels = model$groups)
  actual <- factor(model$groups[model$y], levels = model$groups)
  cm <- table(Actual = actual, Predicted = preds)
  print(cm)
  cat("\n")
  n <- length(actual)
  correct <- sum(preds == actual)
  cat("Training error: ", round((1 - correct / n) * 100, 2), "%\n", sep = "")
}

print_oob_confusion_matrix <- function(model) {
  # `oob_predictions()` is the lazy accessor; returns a factor with `NA` for
  # observations with no OOB tree.
  oob_preds <- oob_predictions(model)
  oob_mask  <- !is.na(oob_preds)
  preds     <- oob_preds[oob_mask]
  actual    <- factor(model$groups[model$y[oob_mask]], levels = model$groups)
  cm <- table(Actual = actual, Predicted = preds)
  print(cm)
  cat("\n")
  n <- length(actual)
  correct <- sum(as.character(preds) == as.character(actual))
  cat("OOB error: ", round((1 - correct / n) * 100, 2), "%\n", sep = "")
}

process_predict_arguments <- function(object, new_data, ...) {
  if (is.null(new_data)) {
    new_data <- list(...)[[1]]
  }

  # Only run model.matrix when `new_data` is a data.frame — that's what
  # model.matrix actually needs. If the caller already handed us a numeric
  # matrix (e.g. `predict(fit, fit$x)` or a hand-prepared design matrix),
  # accept it as-is. Previously this path errored with "data must be a
  # data.frame, not a matrix or an array" whenever the model was fit via
  # the formula interface, which is surprising and blocks any downstream
  # generic that wants to predict on `model$x` (notably `fitted()`).
  if (!is.null(object$formula) && is.data.frame(new_data)) {
    new_data <- model.matrix(object$formula, new_data)
  }

  as.matrix(new_data)
}

resolve_model_data <- function(formula, data, x, y, mode = NULL) {
  if (!is.null(formula) && !is.null(data)) {
    if (!inherits(formula, "formula")) {
      stop("`formula` must be a formula object.")
    }

    if (!is.data.frame(data)) {
      stop("`data` must be a data frame.")
    }

    formula <- update(formula(terms(formula, data = data)), . ~ . - 1)

    y <- model.response(model.frame(formula, data))
    x <- model.matrix(formula, data, response = TRUE)
  } else if (is.null(x) || is.null(y)) {
    stop("For the matrix interface, both `x` and `y` must be provided.")
  }

  x <- as.matrix(x)

  if (!is.numeric(x)) {
    stop("All columns in `x` must be numeric.")
  }

  if (anyNA(x)) {
    stop("`x` must not contain NA or NaN values.")
  }

  # Mode resolution. Three sources of truth, in priority order:
  #
  #   1. Explicit `mode` argument: caller stated their intent. Validate the
  #      response against it — error fast on a real mismatch (e.g. regression
  #      asked for, factor `y` supplied), coerce silently when the response
  #      can satisfy the request (e.g. classification asked for, integer
  #      labels supplied — the common binary-0/1 case).
  #   2. Response type: factor / character → classification, numeric →
  #      regression. The default when `mode` is left `NULL`.
  #
  # Returning the resolved `mode` (alongside the prepared `y` / `groups`)
  # is what the rest of the pipeline keys off: `resolve_strategies(mode = ...)`
  # for default strategies, `ppforest2_train` (which dispatches on `spec$mode`)
  # for the C++ dispatch, and the Rcpp wrap layer for the S3 class.
  if (!is.null(mode)) {
    if (!is.character(mode) || length(mode) != 1L || !mode %in% c("classification", "regression")) {
      stop("`mode` must be either \"classification\" or \"regression\" (or NULL to auto-detect).")
    }
    if (identical(mode, "regression") && is.factor(y)) {
      stop("`mode = \"regression\"` requires a numeric response; got a factor.")
    }
    is_regression <- identical(mode, "regression")
  } else {
    is_regression <- is.numeric(y) && !is.factor(y)
  }

  if (nrow(x) != length(y)) {
    stop("`x` and `y` must have the same number of observations.")
  }

  if (anyNA(y)) {
    stop("`y` must not contain NA or NaN values.")
  }
  if (is_regression && !all(is.finite(y))) {
    stop("`y` must contain only finite values for regression (no Inf / -Inf).")
  }

  if (is_regression) {
    ord <- order(y)
    x_sorted <- x[ord, , drop = FALSE]
    y_sorted <- as.numeric(y[ord])

    return(
      list(
        x = x_sorted,
        y = as.matrix(y_sorted),
        groups = character(0),
        formula = formula,
        mode = "regression"
      )
    )
  }

  # Classification path.
  if (!is.factor(y)) {
    y <- factor(y)
  }

  if (nlevels(y) < 2) {
    stop("`y` must have at least 2 groups.")
  }

  if (nrow(x) < nlevels(y)) {
    stop("Not enough observations: need at least as many rows as groups.")
  }

  return(
    list(
      x = x,
      y = as.matrix(as.numeric(y)),
      groups = levels(y),
      formula = formula,
      mode = "classification"
    )
  )
}
