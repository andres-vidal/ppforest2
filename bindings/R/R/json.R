#' Save a model to a JSON file.
#'
#' Serializes a \code{pptr} or \code{pprf} model to JSON format compatible
#' with the C++ CLI. The JSON includes the model structure, group labels,
#' training parameters, and optionally variable importance metrics.
#'
#' @param model A \code{pptr} or \code{pprf} model (inheriting from \code{ppmodel}).
#' @param path File path to write the JSON to.
#' @param include_metrics If \code{TRUE} (default), include variable importance
#'   and (for forests) OOB error in the output. Set to \code{FALSE} to save
#'   only the model structure and metadata.
#' @param ... Additional arguments (currently unused).
#' @seealso \code{\link{load_json}}, \code{\link{pptr}}, \code{\link{pprf}}
#' @examples
#' model <- pptr(Species ~ ., data = iris, seed = 0)
#' path <- tempfile(fileext = ".json")
#' save_json(model, path)
#'
#' @export
save_json <- function(model, path, ...) {
  UseMethod("save_json")
}

#' @rdname save_json
#' @export
save_json.ppmodel <- function(model, path, include_metrics = TRUE, ...) {
  if (include_metrics && is.null(model$x)) {
    warning("Training data not available; saving without metrics.", call. = FALSE)
    include_metrics <- FALSE
  }
  x <- if (!is.null(model$x)) model$x else matrix(0, nrow = 0, ncol = 0)
  feature_names <- if (!is.null(model$x)) colnames(model$x) else character(0)
  groups <- if (!is.null(model$groups)) model$groups else character(0)

  # `y` is the unified response vector: integer class labels (classification)
  # or the continuous response (regression). Single field, mode-agnostic —
  # matches the C++ unified-y convention.
  y <- if (!is.null(model$y)) model$y else {
    if (identical(model$mode, "regression")) numeric(0) else integer(0)
  }

  json_str <- ppforest2_save_model_json(
    model, groups, include_metrics, x, y, feature_names
  )
  writeLines(json_str, path)
  invisible(path)
}


#' Load a model from a JSON file.
#'
#' Deserializes a \code{pptr} or \code{pprf} model from a JSON file. The model
#' can be used for prediction immediately. If variable importance metrics and
#' OOB fields were saved, they are restored into the model's lazy-computation
#' cache so later accessor calls (\code{oob_error()}, \code{permuted_importance()},
#' etc.) return the stored values without recomputation.
#'
#' Note that \code{formula}, \code{x}, and \code{y} are not stored in the JSON
#' and will be \code{NULL} on the loaded model. Formula-based prediction and
#' any accessor that requires training data will error unless the user
#' re-attaches those fields.
#'
#' @param path File path to read the JSON from.
#' @return A \code{pptr} or \code{pprf} model (with the appropriate
#'   \code{_classification} / \code{_regression} subclass).
#' @seealso \code{\link{save_json}}, \code{\link{pptr}}, \code{\link{pprf}}
#' @examples
#' model <- pptr(Species ~ ., data = iris, seed = 0)
#' path <- tempfile(fileext = ".json")
#' save_json(model, path)
#' loaded <- load_json(path)
#' predict(loaded, as.matrix(iris[, 1:4]))
#'
#' @export
load_json <- function(path) {
  # The underlying C++ loader throws `std::runtime_error` (surfaced by Rcpp
  # as a plain R error without the file path) for any structural problem —
  # malformed JSON, mode/groups mismatch, missing required keys, etc. Wrap
  # here so the user sees which file triggered the failure.
  model <- tryCatch(
    ppforest2_load_model_json(path),
    error = function(e) {
      stop(
        "Failed to load model from '", path, "': ", conditionMessage(e),
        call. = FALSE
      )
    }
  )
  model$groups <- as.character(model$groups)

  # Restore mode from the training spec.
  model$mode <- model$training_spec$mode %||% "classification"

  is_forest     <- !is.null(model$trees)
  is_regression <- identical(model$mode, "regression")

  if (is_forest && !is_regression) {
    for (i in seq_along(model$trees)) {
      model$trees[[i]]$groups <- model$groups
    }
  }

  # Cache env for lazy accessors. Prime with values that travelled through JSON
  # (the C++ save/load path persists OOB metrics and weighted/permuted VI).
  # `oob_metrics` mirrors the C++ Export shape: a list with `confusion_matrix`
  # + `error_rate` (classification) or `mse`/`mae`/`r_squared` (regression).
  # `oob_error()` extracts the right scalar from it without recomputation.
  model$.cache <- .new_cache()
  oob_scalar <- if (!is.null(model$oob_metrics)) {
    if (is_regression) model$oob_metrics$mse else model$oob_metrics$error_rate
  } else NULL
  .prime_cache(model, "oob_error",            oob_scalar)
  .prime_cache(model, "weighted_importance",  model$vi$weighted)
  .prime_cache(model, "permuted_importance",  model$vi$permuted)
  # Clean up primed VI fields so users access via generics, not raw slots.
  # `oob_metrics` itself stays on the model — it's the richer companion to
  # `oob_error()` and `model$training_metrics`.
  model$vi$weighted <- NULL
  model$vi$permuted <- NULL

  # Class is set by the Rcpp wrap layer (see `make_model_class` in
  # bindings/R/inst/include/ppforest2.h), which derives it from
  # `training_spec->mode`. The deserialized model already carries the
  # correct 3-level S3 class — no reassignment needed.

  model
}
