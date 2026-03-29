#' Save a model to a JSON file.
#'
#' Serializes a \code{pptr} or \code{pprf} model to JSON format compatible
#' with the C++ CLI. The JSON includes the model structure, group labels,
#' training parameters, and optionally variable importance metrics.
#'
#' @param model A \code{pptr} or \code{pprf} model.
#' @param path File path to write the JSON to.
#' @param include_metrics If \code{TRUE} (default), include variable importance
#'   and (for forests) OOB error in the output. Set to \code{FALSE} to save
#'   only the model structure and metadata.
#' @param ... Additional arguments (currently unused).
#' @seealso \code{\link{load_json}}, \code{\link{pptr}}, \code{\link{pprf}}
#' @examples
#' model <- pptr(Type ~ ., data = iris, seed = 42)
#' path <- tempfile(fileext = ".json")
#' save_json(model, path)
#'
#' @export
save_json <- function(model, path, ...) {
  UseMethod("save_json")
}

#' @rdname save_json
#' @export
save_json.pptr <- function(model, path, include_metrics = TRUE, ...) {
  if (!is.null(model$x)) {
    n_obs <- nrow(model$x)
    n_features <- ncol(model$x)
    feature_names <- colnames(model$x)
  } else {
    n_obs <- 0L
    n_features <- 0L
    feature_names <- character(0)
  }
  json_str <- ppforest2_save_tree_json(
    model,
    model$groups,
    model$vi,
    model$training_spec,
    model$seed,
    include_metrics,
    n_obs,
    n_features,
    feature_names
  )
  writeLines(json_str, path)
  invisible(path)
}

#' @rdname save_json
#' @export
save_json.pprf <- function(model, path, include_metrics = TRUE, ...) {
  if (!is.null(model$x)) {
    n_obs <- nrow(model$x)
    n_features <- ncol(model$x)
    feature_names <- colnames(model$x)
  } else {
    n_obs <- 0L
    n_features <- 0L
    feature_names <- character(0)
  }
  json_str <- ppforest2_save_forest_json(
    model,
    model$groups,
    model$vi,
    model$training_spec,
    model$seed,
    model$oob_error,
    include_metrics,
    n_obs,
    n_features,
    feature_names
  )
  writeLines(json_str, path)
  invisible(path)
}

#' Load a model from a JSON file.
#'
#' Deserializes a \code{pptr} or \code{pprf} model from a JSON file. The model
#' can be used for prediction immediately. If variable importance metrics were
#' saved, they are restored as well.
#'
#' Note that \code{formula}, \code{x}, and \code{y} are not stored in the JSON
#' and will be \code{NULL} on the loaded model. This means formula-based
#' \code{predict()} (passing a data frame) will not work; pass a numeric matrix
#' instead.
#'
#' @param path File path to read the JSON from.
#' @return A \code{pptr} or \code{pprf} model.
#' @seealso \code{\link{save_json}}, \code{\link{pptr}}, \code{\link{pprf}}
#' @examples
#' model <- pptr(Type ~ ., data = iris, seed = 42)
#' path <- tempfile(fileext = ".json")
#' save_json(model, path)
#' loaded <- load_json(path)
#' predict(loaded, as.matrix(iris[, 1:4]))
#'
#' @export
load_json <- function(path) {
  json_str <- paste(readLines(path, warn = FALSE), collapse = "\n")
  meta <- ppforest2_load_json_meta(json_str)

  if (meta$model_type == "forest") {
    model <- ppforest2_forest_from_json(json_str)
    class(model) <- "pprf"
    model$oob_error <- meta$oob_error

    model$groups <- as.character(meta$groups)
    for (i in seq_along(model$trees)) {
      class(model$trees[[i]]) <- "pptr"
      model$trees[[i]]$groups <- model$groups
    }
  } else {
    model <- ppforest2_tree_from_json(json_str)
    class(model) <- "pptr"
    model$groups <- as.character(meta$groups)
  }

  model$training_spec <- meta$training_spec
  model$seed <- meta$seed
  model$vi <- meta$vi
  model$formula <- NULL
  model$x <- NULL
  model$y <- NULL

  model
}
