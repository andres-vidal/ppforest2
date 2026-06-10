#' @useDynLib ppforest2
#' @importFrom Rcpp evalCpp
#' @importFrom stats model.frame model.matrix model.response formula predict sd terms update fitted residuals nobs
NULL

#' Trains a Random Forest of Project-Pursuit oblique decision trees.
#'
#' This function trains a Random Forest of Project-Pursuit oblique decision tree using either a formula and data frame interface or a matrix-based interface. When using the formula interface, specify the model formula and the data frame containing the variables. For the matrix-based interface, provide matrices for the features and labels directly.
#' The number of trees is controlled by the \code{size} parameter. Each tree is trained on a stratified bootstrap sample drawn from the data.
#' The number of variables to consider at each split is controlled by the \code{n_vars} parameter.
#' If \code{lambda = 0}, the model is trained using Linear Discriminant Analysis (LDA). If \code{lambda > 0}, the model is trained using Penalized Discriminant Analysis (PDA).
#'
#' Mode is taken from the \code{mode} argument when explicit, and otherwise auto-detected from `y` (factor/character → classification, numeric → regression). Pass \code{mode = "classification"} to force classification on integer labels (e.g. binary 0/1), or \code{mode = "regression"} to assert intent on numeric responses.
#'
#' OOB error, OOB predictions, permuted variable importance, and weighted variable importance are computed lazily on first access via the accessor functions (`oob_error()`, `oob_predictions()`, `permuted_importance()`, `weighted_importance()`). Training itself is fast because these OOB-based computations are deferred.
#'
#' @param formula A formula of the form \code{y ~ x1 + x2 + ...}, where \code{y} is a vector of labels and \code{x1}, \code{x2}, ... are the features.
#' @param data A data frame containing the variables in the formula.
#' @param x A matrix containing the features for each observation.
#' @param y A matrix containing the labels for each observation.
#' @param mode Training mode: either \code{"classification"} or \code{"regression"}. When \code{NULL} (default), mode is auto-detected from \code{y}'s type — factor or character vectors trigger classification, numeric vectors trigger regression. Setting it explicitly is useful for the binary-integer-labels case (\code{mode = "classification"} with integer 0/1 labels) and for failing fast on a type mismatch (\code{mode = "regression"} with a factor \code{y} errors immediately).
#' @param size The number of trees in the forest (default: 100).
#' @param lambda A regularization parameter. If \code{lambda = 0}, the model is trained using Linear Discriminant Analysis (LDA). If \code{lambda > 0}, the model is trained using Penalized Discriminant Analysis (PDA). Cannot be used together with \code{pp}.
#' @param n_vars The number of variables to consider at each split (integer). These are chosen uniformly in each split. The default is all variables. Cannot be used together with \code{p_vars} or \code{dr}.
#' @param p_vars The proportion of variables to consider at each split (number between 0 and 1, exclusive). For example, \code{p_vars = 0.5} uses half the features. Cannot be used together with \code{n_vars} or \code{dr}.
#' @param seed An optional integer seed for reproducibility. If \code{NULL} (default), a seed is drawn from R's RNG, so \code{set.seed()} controls reproducibility. If an integer is provided, that value is used directly. The same seed is used for training and for computing permuted variable importance.
#' @param max_retries Maximum number of retries for degenerate trees (default: 3). When a bootstrap sample yields a singular covariance matrix, the tree is retrained with a different seed up to this many times.
#' @param threads The number of threads to use. The default is the number of cores available.
#' @param pp A projection pursuit strategy object created by \code{\link{pp_pda}}. Cannot be used together with \code{lambda}.
#' @param vars A variable selection strategy object created by \code{\link{vars_uniform}} or \code{\link{vars_all}}. Cannot be used together with \code{n_vars} or \code{p_vars}.
#' @param cutpoint A split cutpoint strategy object created by \code{\link{cutpoint_mean_of_means}} (default).
#' @param stop A stopping rule object. Default depends on mode:
#'   \code{\link{stop_pure_node}()} for classification, and
#'   \code{stop_any(stop_min_size(5), stop_min_variance(0.01))} for regression.
#' @param binarize A binarization strategy object. Default depends on mode:
#'   \code{\link{binarize_largest_gap}()} for classification, and
#'   \code{\link{binarize_disabled}()} for regression (regression's default
#'   grouping always yields a 2-group partition, so no binarization is needed).
#' @param grouping A grouping strategy object. Default depends on mode:
#'   \code{\link{grouping_by_label}()} for classification, and
#'   \code{\link{grouping_by_cutpoint}()} for regression.
#' @param leaf A leaf strategy object. Default depends on mode:
#'   \code{\link{leaf_majority_vote}()} for classification, and
#'   \code{\link{leaf_mean_response}()} for regression.
#' @return A \code{pprf} model. Its S3 class vector is
#'   \code{c("pprf_classification", "pprf", "ppmodel")} or
#'   \code{c("pprf_regression", "pprf", "ppmodel")} depending on the mode.
#' @seealso \code{\link{predict.pprf_classification}}, \code{\link{predict.pprf_regression}}, \code{\link{formula.ppmodel}}, \code{\link{oob_error}}, \code{\link{save_json}}, \code{\link{load_json}}, \code{\link{pp_rand_forest}} for parsnip integration, \code{vignette("introduction")} for a tutorial
#' @examples
#'
#' # Example 1: formula interface with the `iris` dataset
#' pprf(Species ~ ., data = iris)
#'
#' # Example 2: formula interface with the `iris` dataset with regularization
#' pprf(Species ~ ., data = iris, lambda = 0.5)
#'
#' # Example 3: matrix interface with the `iris` dataset
#' pprf(x = iris[, 1:4], y = iris[, 5])
#'
#' # Example 4: matrix interface with the `iris` dataset with regularization
#' pprf(x = iris[, 1:4], y = iris[, 5], lambda = 0.5)
#'
#' # Example 5: formula interface with the `crabs` dataset
#' pprf(Type ~ ., data = crabs)
#'
#' # Example 6: formula interface with the `crabs` dataset with regularization
#' pprf(Type ~ ., data = crabs, lambda = 0.5)
#'
#' @export
pprf <- function(
    formula = NULL,
    data = NULL,
    x = NULL,
    y = NULL,
    mode = NULL,
    size = 100,
    lambda = 0,
    n_vars = NULL,
    p_vars = NULL,
    seed = NULL,
    max_retries = 3L,
    threads = NULL,
    pp = NULL,
    vars = NULL,
    cutpoint = NULL,
    stop = NULL,
    binarize = NULL,
    grouping = NULL,
    leaf = NULL) {
  # Capture the call before evaluating anything else so `update()` can
  # rebuild and re-evaluate it with the user's changes substituted in
  # (this is the standard `stats::update.default` contract).
  cl <- match.call()

  if (!is.null(seed) && (!is.numeric(seed) || length(seed) != 1 || seed != as.integer(seed)))
    stop("`seed` must be a single integer or NULL.")

  if (!is.numeric(size) || length(size) != 1 || size < 1 || size != as.integer(size))
    stop("`size` must be a positive integer.")

  if (!is.numeric(max_retries) || length(max_retries) != 1 || max_retries < 0 || max_retries != as.integer(max_retries))
    stop("`max_retries` must be a non-negative integer.")

  if (!is.null(threads) && (!is.numeric(threads) || length(threads) != 1 || threads < 1 || threads != as.integer(threads)))
    stop("`threads` must be a positive integer or NULL.")

  args <- resolve_model_data(formula, data, x, y, mode = mode)

  x <- args$x
  y <- args$y
  groups <- args$groups
  formula <- args$formula
  mode <- args$mode

  strategies <- resolve_strategies(
    pp = pp, lambda = lambda, lambda_missing = missing(lambda),
    vars = vars, n_vars = n_vars, n_vars_missing = missing(n_vars),
    p_vars = p_vars, p_vars_missing = missing(p_vars),
    cutpoint = cutpoint, stop = stop, binarize = binarize, grouping = grouping,
    leaf = leaf, default_vars = vars_uniform(),
    n_features = ncol(x))

  if (is.null(seed)) {
    seed <- sample.int(.Machine$integer.max, 1L)
  }

  effective_threads <- if (is.null(threads)) parallel::detectCores() else threads
  if (effective_threads > 1 && !ppforest2_has_openmp()) {
    warning("OpenMP is not available. The forest will be trained using a single thread.\n",
            "On macOS, install libomp: brew install libomp", call. = FALSE)
  }

  training_spec <- list(
    pp = strategies$pp,
    vars = strategies$vars,
    cutpoint = strategies$cutpoint,
    stop = strategies$stop,
    binarize = strategies$binarize,
    grouping = strategies$grouping,
    leaf = strategies$leaf,
    mode = mode,
    size = as.integer(size),
    seed = as.integer(seed),
    threads = if (is.null(threads)) 0L else as.integer(threads),
    max_retries = as.integer(max_retries))

  # `ppforest2_train` is mode-aware on the C++ side: it dispatches on
  # `training_spec$mode` and applies the appropriate index decode + sort.
  model <- ppforest2_train(training_spec, x, y)

  if (isTRUE(model$degenerate)) {
    warning("Some splits could not separate groups (degenerate nodes). ",
            "This can be caused by ill-conditioned variables in the input data, ",
            "or by bootstrap samples that produce singular covariance matrices. ",
            "Consider reviewing your data or adjusting `max_retries` (currently ", max_retries, "). ",
            "Degenerate nodes predict the group with the most observations. ",
            "Degenerate trees are excluded from variable importance calculations.",
            call. = FALSE)
  }

  # Regression models have no groups; this loop only applies to classification.
  if (!identical(mode, "regression")) {
    for (i in 1:size) {
      model$trees[[i]]$groups <- groups
    }
  }

  model$call    <- cl
  model$seed    <- seed
  model$groups  <- groups
  model$formula <- formula
  model$mode    <- mode
  model$x       <- x
  model$y       <- y

  # Cheap, always-available VI fields. Expensive OOB-based importances
  # (`permuted_importance()`, `weighted_importance()`) compute lazily.
  scale <- apply(x, 2, sd)
  scale[scale == 0] <- 1
  model$vi <- list(
    scale       = scale,
    projections = ppforest2_vi_projections_forest(model, ncol(x), scale)
  )

  # Lazy-compute cache for OOB metrics and permuted/weighted importance.
  model$.cache <- .new_cache()

  # Class is set by the Rcpp wrap layer (see `make_model_class` in
  # bindings/R/inst/include/ppforest2.h), which derives it from
  # `forest.training_spec->mode`. Don't reassign here — that would let the
  # R-side guess drift away from the C++ truth.

  model
}


# ---------------------------------------------------------------------------
# Prediction: split per mode.
# ---------------------------------------------------------------------------

#' Predicts labels or vote proportions from a pprf model (classification mode).
#'
#' @param object A \code{pprf_classification} model.
#' @param new_data A data frame or matrix of new observations. If \code{NULL}, the first positional argument in \code{...} is used for backward compatibility.
#' @param type The type of prediction: \code{"class"} (default) returns a factor of predicted labels, \code{"prob"} returns a data frame of vote proportions.
#' @param ... For backward compatibility, the first positional argument is treated as \code{new_data} when \code{new_data} is \code{NULL}.
#' @return If \code{type = "class"}, a factor of predicted labels. If \code{type = "prob"}, a data frame with one column per group, each row summing to 1.
#' @seealso \code{\link{pprf}}, \code{\link{predict.pprf_regression}}
#' @examples
#' model <- pprf(Species ~ ., data = iris)
#' predict(model, iris)
#' predict(model, iris, type = "prob")
#' @export
predict.pprf_classification <- function(object, new_data = NULL, type = NULL, ...) {
  x <- process_predict_arguments(object, new_data, ...)
  if (is.null(type)) type <- "class"

  if (type == "prob") {
    probs <- ppforest2_predict_forest_prob(object, x)
    df <- as.data.frame(probs)
    colnames(df) <- object$groups
    return(df)
  }

  if (type != "class") {
    stop("`type = \"", type, "\"` is not supported for classification models. ",
         "Use \"class\" (default) or \"prob\".", call. = FALSE)
  }

  y <- ppforest2_predict_forest(object, x)
  as.factor(object$groups[y])
}

#' Predicts numeric responses from a pprf model (regression mode).
#'
#' @param object A \code{pprf_regression} model.
#' @param new_data A data frame or matrix of new observations.
#' @param type Must be \code{"response"} (default).
#' @param ... For backward compatibility, the first positional argument is treated as \code{new_data} when \code{new_data} is \code{NULL}.
#' @return A numeric vector of mean predictions across the forest's trees.
#' @seealso \code{\link{pprf}}, \code{\link{predict.pprf_classification}}
#' @export
predict.pprf_regression <- function(object, new_data = NULL, type = NULL, ...) {
  x <- process_predict_arguments(object, new_data, ...)
  if (is.null(type)) type <- "response"

  if (type %in% c("class", "prob")) {
    stop("`type = \"", type, "\"` is not available for regression models. ",
         "Use `type = \"response\"`.", call. = FALSE)
  }

  if (type != "response") {
    stop("`type = \"", type, "\"` is not recognised. Use \"response\".", call. = FALSE)
  }

  as.numeric(ppforest2_predict_forest(object, x))
}


# ---------------------------------------------------------------------------
# print.pprf -- minimal, mode-agnostic. See `summary()` for the full breakdown.
# ---------------------------------------------------------------------------

#' Prints a compact summary of a pprf forest.
#' @param x A \code{pprf} model.
#' @param ... Unused.
#' @seealso \code{\link{summary.pprf}}
#' @export
print.pprf <- function(x, ...) {
  cat("\n")
  cat("Random Forest of Project-Pursuit Oblique Decision Trees\n")
  if (!is.null(x$call)) {
    cat("  Call:        ", paste(deparse(x$call, width.cutoff = 80L), collapse = "\n               "), "\n", sep = "")
  }
  cat("  Trees:       ", length(x$trees), "\n", sep = "")
  cat("  Mode:        ", x$mode, "\n", sep = "")
  if (!is.null(x$groups) && length(x$groups) > 0L) {
    cat("  Group names: ", paste(x$groups, collapse = ", "), "\n", sep = "")
  }
  if (!is.null(x$formula)) {
    cat("  Formula:     ", deparse(x$formula), "\n", sep = "")
  }
  cat("\n")
  invisible(x)
}


# ---------------------------------------------------------------------------
# summary -- layered via NextMethod:
#   summary.pprf_classification / summary.pprf_regression
#     -> summary.pprf (forest-level header, VI table)
#       -> summary.ppmodel (data summary, config, formula)
# ---------------------------------------------------------------------------

#' @export
summary.ppmodel <- function(object, ...) {
  model <- object
  cat("\n")
  cat("Data Summary:\n")
  cat("  observations:", nrow(model$x), "\n")
  cat("  features:    ", ncol(model$x), "\n")
  if (identical(model$mode, "classification")) {
    cat("  groups:      ", length(model$groups), "\n")
    cat("  group names: ", paste(model$groups, collapse = ", "), "\n")
  }
  if (!is.null(model$formula)) {
    cat("  formula:     ", deparse(model$formula), "\n")
  }
  cat("\n")
  invisible(model)
}

#' Summary of a pprf forest (shared header + VI).
#'
#' @param object A \code{pprf} model.
#' @param ... Unused.
#' @export
summary.pprf <- function(object, ...) {
  model <- object
  if (is.null(model$x)) {
    cat("\n(Empty pprf model -- no training data available.)\n")
    return(invisible(model))
  }

  cat("\n")
  cat(if (identical(model$mode, "regression")) {
    "Random Forest of Project-Pursuit Oblique Regression Trees\n"
  } else {
    "Random Forest of Project-Pursuit Oblique Decision Trees\n"
  })
  cat("\n")
  cat("Size:", length(model$trees), "trees\n")
  print_training_spec(model$training_spec)

  NextMethod()  # summary.ppmodel -- data summary block

  invisible(model)
}

#' @export
summary.pprf_classification <- function(object, ...) {
  # Let the shared scaffolding (pprf -> ppmodel) run first.
  NextMethod()
  model <- object

  cat("Training Confusion Matrix:\n\n")
  print_confusion_matrix(ppforest2_predict_forest(model, model$x), model)
  cat("\n")
  cat("OOB Confusion Matrix:\n\n")
  print_oob_confusion_matrix(model)
  cat("\n")

  .print_vi_table(model, include_oob_importances = TRUE)
  invisible(model)
}

#' @export
summary.pprf_regression <- function(object, ...) {
  NextMethod()
  model <- object

  # Training metrics
  preds <- ppforest2_predict_forest(model, model$x)
  y <- model$y
  mse <- mean((preds - y)^2)
  mae <- mean(abs(preds - y))
  ss_tot <- sum((y - mean(y))^2)
  r2 <- if (ss_tot > 0) 1 - sum((preds - y)^2) / ss_tot else 0
  cat("Training Metrics:\n")
  cat("  MSE:", format(mse, nsmall = 6), "\n")
  cat("  MAE:", format(mae, nsmall = 6), "\n")
  cat("  R\u00b2: ", format(r2, nsmall = 6), "\n\n")

  oob <- oob_error(model)
  if (!is.na(oob)) {
    cat("OOB MSE:", format(oob, nsmall = 6), "\n\n")
  } else {
    cat("OOB MSE: not available (no observation has any out-of-bag tree)\n\n")
  }

  .print_vi_table(model, include_oob_importances = TRUE)
  invisible(model)
}


# ---------------------------------------------------------------------------
# Helpers -- these are for `summary()`, not part of the public API.
# ---------------------------------------------------------------------------

# Print a VI table. For forests, `include_oob_importances` pulls the lazy
# `weighted` and `permuted` importances. For trees, these are absent.
.print_vi_table <- function(model, include_oob_importances) {
  cat("Variable Importance:\n\n")

  p <- length(model$vi$projections)
  vnames <- if (!is.null(colnames(model$x))) colnames(model$x) else paste0("x", seq_len(p))
  ord <- order(model$vi$projections, decreasing = TRUE)

  cols <- list(
    Variable   = vnames[ord],
    sigma      = model$vi$scale[ord],
    Projection = model$vi$projections[ord]
  )

  if (include_oob_importances) {
    cols$Weighted <- weighted_importance(model)[ord]
    cols$Permuted <- permuted_importance(model)[ord]
  }

  tbl <- do.call(data.frame, c(cols, list(row.names = seq_len(p))))
  names(tbl)[2] <- "\u03c3"
  print(tbl)

  if (!all(model$vi$scale == 1)) {
    cat("\nNote: Variable importance was calculated using scaled coefficients (|a_j| * \u03c3_j).\n")
    cat("Variable contributions can only be theoretically interpreted as such\n")
    cat("if the model was trained on scaled data. Scaling also changes the\n")
    cat("projection-pursuit optimization, which may affect the resulting tree.\n")
  }
}
