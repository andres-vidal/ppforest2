#' @useDynLib ppforest2
#' @importFrom Rcpp evalCpp
#' @importFrom stats model.frame model.matrix model.response formula predict sd terms update fitted residuals nobs
NULL

#' Trains a Projection-Pursuit oblique decision tree.
#'
#' This function trains a Projection-Pursuit oblique decision tree using either a formula and data frame interface or a matrix-based interface. When using the formula interface, specify the model formula and the data frame containing the variables. For the matrix-based interface, provide matrices for the features and labels directly.
#' If \code{lambda = 0}, the model is trained using Linear Discriminant Analysis (LDA). If \code{lambda > 0}, the model is trained using Penalized Discriminant Analysis (PDA).
#'
#' Mode is taken from the \code{mode} argument when explicit, and otherwise auto-detected from `y` (factor/character → classification, numeric → regression). Pass \code{mode = "classification"} to force classification on integer labels (e.g. binary 0/1), or \code{mode = "regression"} to assert intent on numeric responses.
#'
#' @param formula A formula of the form \code{y ~ x1 + x2 + ...}, where \code{y} is a vector of labels and \code{x1}, \code{x2}, ... are the features.
#' @param data A data frame containing the variables in the formula.
#' @param x A matrix containing the features for each observation.
#' @param y A matrix containing the labels for each observation.
#' @param mode Training mode: either \code{"classification"} or \code{"regression"}. When \code{NULL} (default), mode is auto-detected from \code{y}'s type — factor or character vectors trigger classification, numeric vectors trigger regression. Setting it explicitly is useful for the binary-integer-labels case (\code{mode = "classification"} with integer 0/1 labels) and for failing fast on a type mismatch (\code{mode = "regression"} with a factor \code{y} errors immediately).
#' @param lambda A regularization parameter (default: 0.5). If \code{lambda = 0}, the model is trained using Linear Discriminant Analysis (LDA). If \code{lambda > 0}, the model is trained using Penalized Discriminant Analysis (PDA). The default uses PDA because pure LDA (\code{lambda = 0}) is ill-conditioned when there are more variables than effective observations (see the "Known limitations" section of the README). Cannot be used together with \code{pp}.
#' @param seed An optional integer seed for reproducibility. If \code{NULL} (default), a seed is drawn from R's RNG, so \code{set.seed()} controls reproducibility. If an integer is provided, that value is used directly.
#' @param pp A projection pursuit strategy object created by \code{\link{pp_pda}}. Cannot be used together with \code{lambda}.
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
#' @return A \code{pptr} model. Its S3 class vector is
#'   \code{c("pptr_classification", "pptr", "ppmodel")} or
#'   \code{c("pptr_regression", "pptr", "ppmodel")} depending on the mode.
#' @seealso \code{\link{predict.pptr_classification}}, \code{\link{predict.pptr_regression}}, \code{\link{formula.ppmodel}}, \code{\link{print.pptr}}, \code{\link{save_json}}, \code{\link{load_json}}, \code{\link{pp_tree}} for parsnip integration
#' @examples
#'
#' # Example 1: formula interface with the `iris` dataset
#' pptr(Species ~ ., data = iris)
#'
#' # Example 2: formula interface with the `iris` dataset with regularization
#' pptr(Species ~ ., data = iris, lambda = 0.5)
#'
#' # Example 3: matrix interface with the `iris` dataset
#' pptr(x = iris[, 1:4], y = iris[, 5])
#'
#' @export
pptr <- function(
    formula = NULL,
    data = NULL,
    x = NULL,
    y = NULL,
    mode = NULL,
    lambda = 0.5,
    seed = NULL,
    pp = NULL,
    cutpoint = NULL,
    stop = NULL,
    binarize = NULL,
    grouping = NULL,
    leaf = NULL) {
  # See the matching comment in `pprf()` — capture the call up front so
  # `update()` can rebuild and re-evaluate it.
  cl <- match.call()

  if (!is.null(seed) && (!is.numeric(seed) || length(seed) != 1 || seed != as.integer(seed)))
    stop("`seed` must be a single integer or NULL.")

  args <- resolve_model_data(formula, data, x, y, mode = mode)
  mode <- args$mode

  strategies <- resolve_strategies(
    pp = pp, lambda = lambda, lambda_missing = missing(lambda),
    cutpoint = cutpoint, stop = stop, binarize = binarize, grouping = grouping,
    leaf = leaf)

  x <- args$x
  y <- args$y
  groups <- args$groups
  formula <- args$formula

  if (is.null(seed)) {
    seed <- sample.int(.Machine$integer.max, 1L)
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
    size = 0L,
    seed = as.integer(seed),
    threads = 0L,
    max_retries = 3L)

  # `ppforest2_train` is mode-aware on the C++ side: it dispatches on
  # `training_spec$mode` and applies the appropriate index decode + sort.
  model <- ppforest2_train(training_spec, args$x, args$y)

  if (isTRUE(model$degenerate)) {
    warning("Some splits could not separate groups (degenerate nodes). ",
            "This can be caused by ill-conditioned variables in the input data. ",
            "Degenerate nodes predict the group with the most observations.",
            call. = FALSE)
  }

  model$call    <- cl
  model$seed    <- seed
  model$groups  <- groups
  model$formula <- formula
  model$mode    <- mode
  model$x       <- x
  model$y       <- y

  scale <- apply(x, 2, sd)
  scale[scale == 0] <- 1

  model$vi <- list(
    scale       = scale,
    projections = ppforest2_vi_projections_tree(model, ncol(x), scale)
  )

  model$.cache <- .new_cache()

  # Class is set by the Rcpp wrap layer (see `make_model_class` in
  # bindings/R/inst/include/ppforest2.h), which derives it from
  # `tree.training_spec->mode`. Don't reassign here — that would let the
  # R-side guess drift away from the C++ truth.

  model
}


# ---------------------------------------------------------------------------
# Prediction: split per mode.
# ---------------------------------------------------------------------------

#' Predicts labels or per-group one-hot proportions from a pptr model (classification mode).
#'
#' @param object A \code{pptr_classification} model.
#' @param new_data A data frame or matrix of new observations. If \code{NULL}, the first positional argument in \code{...} is used for backward compatibility.
#' @param type \code{"class"} (default) returns a factor of predicted labels; \code{"prob"} returns a data frame with 1.0 for the predicted group and 0.0 elsewhere.
#' @param ... Backward-compat positional `new_data`.
#' @return A factor or data frame.
#' @seealso \code{\link{pptr}}, \code{\link{predict.pptr_regression}}
#' @export
predict.pptr_classification <- function(object, new_data = NULL, type = NULL, ...) {
  x <- process_predict_arguments(object, new_data, ...)
  if (is.null(type)) type <- "class"

  if (type == "prob") {
    probs <- ppforest2_predict_tree_prob(object, x)
    df <- as.data.frame(probs)
    colnames(df) <- object$groups
    return(df)
  }

  if (type != "class") {
    stop("`type = \"", type, "\"` is not supported for classification trees. ",
         "Use \"class\" (default) or \"prob\".", call. = FALSE)
  }

  y <- ppforest2_predict_tree(object, x)
  as.factor(object$groups[y])
}

#' Predicts numeric responses from a pptr model (regression mode).
#'
#' @param object A \code{pptr_regression} model.
#' @param new_data A data frame or matrix of new observations.
#' @param type Must be \code{"response"} (default).
#' @param ... Backward-compat positional `new_data`.
#' @return A numeric vector.
#' @seealso \code{\link{pptr}}, \code{\link{predict.pptr_classification}}
#' @export
predict.pptr_regression <- function(object, new_data = NULL, type = NULL, ...) {
  x <- process_predict_arguments(object, new_data, ...)
  if (is.null(type)) type <- "response"

  if (type %in% c("class", "prob")) {
    stop("`type = \"", type, "\"` is not available for regression models. ",
         "Use `type = \"response\"`.", call. = FALSE)
  }

  if (type != "response") {
    stop("`type = \"", type, "\"` is not recognised. Use \"response\".", call. = FALSE)
  }

  as.numeric(ppforest2_predict_tree(object, x))
}


# ---------------------------------------------------------------------------
# print.pptr -- tree structure. Leaf label formatting is mode-specific via
# the `print_node` generic dispatched on the model's class.
# ---------------------------------------------------------------------------

#' Prints the structure of a pptr tree.
#' @param x A \code{pptr} model.
#' @param ... Unused.
#' @export
print.pptr <- function(x, ...) {
  cat("\n")
  if (!is.null(x$call)) {
    cat("Call: ", paste(deparse(x$call, width.cutoff = 80L), collapse = "\n      "), "\n\n", sep = "")
  }
  cat("Projection-Pursuit Oblique Decision Tree:\n")
  print_node(x, x$root)
  cat("\n")
  invisible(x)
}

# Internal generic: walks the tree recursively, dispatching on the model's
# class to format leaf values differently per mode.
print_node <- function(model, node, depth = 0) UseMethod("print_node")

#' @export
print_node.pptr_classification <- function(model, node, depth = 0) {
  .print_node_impl(model, node, depth, function(value) model$groups[value])
}

#' @export
print_node.pptr_regression <- function(model, node, depth = 0) {
  .print_node_impl(model, node, depth, function(value) format(as.numeric(value), digits = 4))
}

# Shared recursion. `format_leaf` turns the raw leaf value into a display string.
.print_node_impl <- function(model, node, depth, format_leaf) {
  indent <- paste(rep(" ", depth), collapse = "")

  if (!is.null(node$value)) {
    cat(indent, "Predict:", format_leaf(node$value), "\n")
    return(invisible(NULL))
  }

  projection_str <- paste(
    "[", paste(round(node$projector, 2), collapse = " "), "] * x",
    collapse = ""
  )

  cat(indent, "If (", projection_str, ") < ", node$cutpoint, ":\n", sep = "")

  if (!is.null(node$lower)) {
    print_node(model, node$lower, depth + 1)
  }

  cat(indent, "Else:\n", sep = "")

  if (!is.null(node$upper)) {
    print_node(model, node$upper, depth + 1)
  }
}


# ---------------------------------------------------------------------------
# summary -- layered via NextMethod:
#   summary.pptr_classification / summary.pptr_regression
#     -> summary.pptr (tree-level header + VI table)
#       -> summary.ppmodel (data summary block)
# ---------------------------------------------------------------------------

#' @export
summary.pptr <- function(object, ...) {
  model <- object
  if (is.null(model$x)) {
    cat("\n(Empty pptr model -- no training data available.)\n")
    return(invisible(model))
  }

  cat("\n")
  cat(if (identical(model$mode, "regression")) {
    "Projection-Pursuit Oblique Regression Tree\n"
  } else {
    "Projection-Pursuit Oblique Decision Tree\n"
  })
  cat("\n")
  print_training_spec(model$training_spec)

  NextMethod()  # summary.ppmodel

  invisible(model)
}

#' @export
summary.pptr_classification <- function(object, ...) {
  NextMethod()
  model <- object

  cat("Confusion Matrix:\n\n")
  print_confusion_matrix(ppforest2_predict_tree(model, model$x), model)
  cat("\n")

  .print_vi_table(model, include_oob_importances = FALSE)
  invisible(model)
}

#' @export
summary.pptr_regression <- function(object, ...) {
  NextMethod()
  model <- object

  preds <- ppforest2_predict_tree(model, model$x)
  y <- model$y
  mse <- mean((preds - y)^2)
  mae <- mean(abs(preds - y))
  ss_tot <- sum((y - mean(y))^2)
  r2 <- if (ss_tot > 0) 1 - sum((preds - y)^2) / ss_tot else 0
  cat("Training Metrics:\n")
  cat("  MSE:", format(mse, nsmall = 6), "\n")
  cat("  MAE:", format(mae, nsmall = 6), "\n")
  cat("  R\u00b2: ", format(r2, nsmall = 6), "\n\n")

  .print_vi_table(model, include_oob_importances = FALSE)
  invisible(model)
}
