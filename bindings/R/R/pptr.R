#' @useDynLib ppforest2
#' @importFrom Rcpp evalCpp
#' @importFrom stats model.frame model.matrix model.response formula predict
NULL

#' Trains a Project-Pursuit oblique decision tree.
#'
#' This function trains a Project-Pursuit oblique decision tree using either a formula and data frame interface or a matrix-based interface. When using the formula interface, specify the model formula and the data frame containing the variables. For the matrix-based interface, provide matrices for the features and labels directly.
#' If \code{lambda = 0}, the model is trained using Linear Discriminant Analysis (LDA). If \code{lambda > 0}, the model is trained using Penalized Discriminant Analysis (PDA).
#'
#' @param formula A formula of the form \code{y ~ x1 + x2 + ...}, where \code{y} is a vector of labels and \code{x1}, \code{x2}, ... are the features.
#' @param data A data frame containing the variables in the formula.
#' @param x A matrix containing the features for each observation.
#' @param y A matrix containing the labels for each observation.
#' @param lambda A regularization parameter. If \code{lambda = 0}, the model is trained using Linear Discriminant Analysis (LDA). If \code{lambda > 0}, the model is trained using Penalized Discriminant Analysis (PDA).
#' @param seed An optional integer seed for reproducibility. If \code{NULL} (default), a seed is drawn from R's RNG, so \code{set.seed()} controls reproducibility. If an integer is provided, that value is used directly.
#' @return A pptr model trained on \code{x} and \code{y}.
#' @seealso \code{\link{predict.pptr}}, \code{\link{formula.pptr}}, \code{\link{summary.pptr}}, \code{\link{print.pptr}}, \code{\link{save_json}}, \code{\link{load_json}}, \code{\link{pp_tree}} for parsnip integration, \code{vignette("introduction")} for a tutorial
#' @examples
#'
#' # Example 1: formula interface with the `iris` dataset
#' pptr(Type ~ ., data = iris)
#'
#' # Example 2: formula interface with the `iris` dataset with regularization
#' pptr(Type ~ ., data = iris, lambda = 0.5)
#'
#' # Example 3: matrix interface with the `iris` dataset
#' pptr(x = iris[, 1:4], y = iris[, 5])
#'
#' # Example 4: matrix interface with the `iris` dataset with regularization
#' pptr(x = iris[, 1:4], y = iris[, 5], lambda = 0.5)
#'
#' # Example 5: formula interface with the `crabs` dataset
#' pptr(Type ~ ., data = crabs)
#'
#' # Example 6: formula interface with the `crabs` dataset with regularization
#' pptr(Type ~ ., data = crabs, lambda = 0.5)
#'
#' @export
pptr <- function(
    formula = NULL,
    data = NULL,
    x = NULL,
    y = NULL,
    lambda = 0,
    seed = NULL) {
  if (!is.numeric(lambda) || length(lambda) != 1 || lambda < 0 || lambda > 1)
    stop("`lambda` must be a single number between 0 and 1.")

  if (!is.null(seed) && (!is.numeric(seed) || length(seed) != 1 || seed != as.integer(seed)))
    stop("`seed` must be a single integer or NULL.")

  args <- process_model_arguments(formula, data, x, y)

  x <- args$x
  y <- args$y
  groups <- args$groups
  formula <- args$formula

  if (is.null(seed)) {
    seed <- sample.int(.Machine$integer.max, 1L)
  }

  model <- ppforest2_train_tree_pda(args$x, args$y, lambda, seed)

  if (isTRUE(model$degenerate)) {
    warning("Some splits could not separate groups (degenerate nodes). ",
            "This can be caused by ill-conditioned variables in the input data. ",
            "Degenerate nodes predict the group with the most observations.",
            call. = FALSE)
  }

  class(model) <- "pptr"
  model$seed <- seed
  model$groups <- groups
  model$formula <- formula
  model$x <- x
  model$y <- y

  scale <- apply(x, 2, sd)
  scale[scale == 0] <- 1

  model$vi <- list(
    scale       = scale,
    projections = ppforest2_vi_projections_tree(model, ncol(x), scale)
  )

  model
}

#' Predicts the labels or group indicators of a set of observations using a pptr model.
#'
#' @param object A pptr model.
#' @param new_data A data frame or matrix of new observations to predict. If \code{NULL}, the first positional argument in \code{...} is used for backward compatibility.
#' @param type The type of prediction: \code{"class"} (default) returns a factor of predicted labels, \code{"prob"} returns a data frame with 1.0 for the predicted group and 0.0 elsewhere.
#' @param ... For backward compatibility, the first positional argument is treated as \code{new_data} when \code{new_data} is \code{NULL}.
#' @return If \code{type = "class"}, a factor of predicted labels. If \code{type = "prob"}, a data frame with one column per group.
#' @seealso \code{\link{pptr}} for training, \code{\link{formula.pptr}}, \code{\link{summary.pptr}}
#' @examples
#' # Example 1: with the `iris` dataset
#' model <- pptr(Type ~ ., data = iris)
#' predict(model, iris)
#'
#' # Example 2: with the `crabs` dataset
#' model <- pptr(Type ~ ., data = crabs)
#' predict(model, crabs)
#' 
#' # Example 3: vote proportions
#' model <- pptr(Type ~ ., data = iris)
#' predict(model, iris, type = "prob")
#'
#' @export
predict.pptr <- function(object, new_data = NULL, type = "class", ...) {
  x <- process_predict_arguments(object, new_data, ...)

  if (type == "prob") {
    probs <- ppforest2_predict_tree_prob(object, x)
    df <- as.data.frame(probs)
    colnames(df) <- object$groups
    return(df)
  }

  y <- ppforest2_predict_tree(object, x)
  as.factor(object$groups[y])
}

#' Extracts the formula used to train a pptr model.
#'
#' @param x A pptr model.
#' @param ... (unused) other parameters typically passed to formula.
#' @return The formula used to train the model.
#' @seealso \code{\link{pptr}} for training, \code{\link{predict.pptr}}, \code{\link{summary.pptr}}
#' @examples
#' model <- pptr(Type ~ ., data = iris)
#' formula(model)
#' @export
formula.pptr <- function(x, ...) {
  x$formula
}

print_node <- function(model, node, depth = 0) {
  indent <- paste(rep(" ", depth), collapse = "")

  if (!is.null(node$value)) {
    cat(indent, "Predict:", model$groups[node$value], "\n")
  } else {
    projection_str <- paste(
      "[", paste(round(node$projector, 2), collapse = " "), "] * x",
      collapse = ""
    )

    cat(
      indent,
      "If (", projection_str, ") < ", node$threshold, ":\n",
      sep = ""
    )

    if (!is.null(node$lower)) {
      print_node(model, node$lower, depth + 1)
    }

    cat(indent, "Else:\n", sep = "")
    if (!is.null(node$upper)) {
      print_node(model, node$upper, depth + 1)
    }
  }
}

#' Prints a pptr model.
#' @param x A pptr model.
#' @param ... (unused) other parameters typically passed to print.
#' @seealso \code{\link{pptr}}, \code{\link{summary.pptr}}
#' @examples
#' model <- pptr(Type ~ ., data = iris)
#' print(model)
#'
#' @export
print.pptr <- function(x, ...) {
  model <- x
  if (!is.null(formula(x))) {
    cat("\n")
    cat("Project-Pursuit Oblique Decision Tree:\n")
  }

  print_node(model, model$root)
  cat("\n")
}

#' Summarizes a pptr model.
#' @param object A pptr model.
#' @param ... (unused) other parameters typically passed to summary.
#' @seealso \code{\link{pptr}}, \code{\link{predict.pptr}}, \code{\link{print.pptr}}
#' @examples
#' model <- pptr(Type ~ ., data = iris)
#' summary(model)
#'
#' @export
summary.pptr <- function(object, ...) {
  model <- object

  if (!is.null(model$x)) {
    cat("\n")
    cat("Project-Pursuit Oblique Decision Tree\n")
    cat("\n")
    cat("Regularization parameter:", model$training_spec$lambda, "\n")
    cat("\n")
    cat("Data Summary:\n")
    cat("  observations:", nrow(model$x), "\n")
    cat("  features:    ", ncol(model$x), "\n")
    cat("  groups:      ", length(model$groups), "\n")
    cat("  group names: ", paste(model$groups, collapse = ", "), "\n")
    if (!is.null(model$formula)) {
      cat("  formula:     ", deparse(model$formula), "\n")
    }
    cat("\n")
    cat("Confusion Matrix:\n\n")
    print_confusion_matrix(ppforest2_predict_tree(model, model$x), model)
    cat("\n")
    cat("Variable Importance:\n\n")
    p <- length(model$vi$projections)
    vnames <- if (!is.null(colnames(model$x))) colnames(model$x) else paste0("x", seq_len(p))
    ord <- order(model$vi$projections, decreasing = TRUE)
    tbl <- data.frame(
      Variable   = vnames[ord],
      sigma      = model$vi$scale[ord],
      Projection = model$vi$projections[ord],
      row.names  = seq_len(p)
    )
    names(tbl)[2] <- "\u03c3"
    print(tbl)
    if (!all(model$vi$scale == 1)) {
      cat("\nNote: Variable importance was calculated using scaled coefficients (|a_j| * \u03c3_j).\n")
      cat("Variable contributions can only be theoretically interpreted as such\n")
      cat("if the model was trained on scaled data. Scaling also changes the\n")
      cat("projection-pursuit optimization, which may affect the resulting tree.\n")
    }
  }
  cat("\n")
}
