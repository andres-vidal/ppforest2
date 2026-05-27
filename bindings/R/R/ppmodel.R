#' @useDynLib ppforest2
#' @importFrom Rcpp evalCpp
NULL

# ---------------------------------------------------------------------------
# Memoization cache (environment-based; mutable across S3 method calls).
#
# Models are S3 lists (copy-on-modify), which can't memoize via field writes.
# Each model carries an environment `$.cache` used as a scratchpad by the
# accessor methods below. `load_json` populates the cache with values that
# were persisted during training-time save.
# ---------------------------------------------------------------------------

# Create a fresh cache environment.
.new_cache <- function() new.env(parent = emptyenv())

# Return the cached value for `key`, computing via `compute_fn` on first access.
# Falls back to uncached compute if `model$.cache` is missing (e.g. models
# assembled manually in tests).
.cached_or_compute <- function(model, key, compute_fn) {
  cache <- model$.cache
  if (is.null(cache)) return(compute_fn())
  if (!exists(key, envir = cache, inherits = FALSE)) {
    assign(key, compute_fn(), envir = cache)
  }
  get(key, envir = cache, inherits = FALSE)
}

# Stash a pre-computed value directly into the cache (used by load_json to
# preserve OOB metrics saved during training-time serialization).
.prime_cache <- function(model, key, value) {
  cache <- model$.cache
  if (is.null(cache)) return(invisible(NULL))
  if (!is.null(value)) assign(key, value, envir = cache)
  invisible(NULL)
}


# ---------------------------------------------------------------------------
# Public OOB accessors (generics + methods).
#
# Only forests have OOB concepts; calling these on a `pptr` model errors.
# The accessors compute lazily using the training data stored on the model
# (`$x`, `$y` — `$y` is integer class labels for classification and the
# continuous response for regression) and cache the result in `$.cache`.
# ---------------------------------------------------------------------------

#' Out-of-bag error for a random forest.
#'
#' Computes (or returns the cached) OOB error using the training data stored
#' on the model. For classification, this is the misclassification rate in
#' `[0, 1]`. For regression, it is the mean squared error against the
#' continuous response.
#'
#' @param model A \code{pprf} forest model.
#' @return A numeric scalar in `[0, 1]` for classification or `[0, Inf)` for
#'   regression. Returns `NA_real_` when no observation has any out-of-bag
#'   tree (e.g. a degenerate forest where every tree saw every row). Callers
#'   should check with `is.na()` rather than comparing against a sentinel
#'   value; in earlier versions this condition was signalled as `-1`, which
#'   was not distinguishable from a (mathematically impossible but
#'   representable) error rate.
#' @seealso \code{\link{oob_predictions}}, \code{\link{oob_samples}}
#' @export
oob_error <- function(model) UseMethod("oob_error")

#' @export
oob_error.default <- function(model) {
  stop(
    "`oob_error()` is only defined for `pprf` forest models, not objects of class '",
    paste(class(model), collapse = "/"), "'.",
    call. = FALSE
  )
}

# Rcpp returns a length-1 `NumericVector` from the C++ OOB-error bindings.
# That already behaves as a length-1 R numeric for every operation callers
# care about (comparisons, `is.na()`, arithmetic, assignment), so no
# `as.numeric()` coercion is needed on either branch. The length-1 vector
# carries either the error value or `NA_real_` (never the legacy `-1`
# sentinel — the C++ side translates `std::nullopt` at the boundary).

#' @export
oob_error.pprf <- function(model) {
  # Mode dispatch lives in `ppforest2_oob_error` on the C++ side: it
  # decodes `y` from R 1-based factor codes only for classification,
  # and computes either misclassification rate or MSE based on the
  # forest's `training_spec$mode`. Both subclasses can share this
  # method as a result — no per-mode wrapper needed at the R level.
  .cached_or_compute(model, "oob_error", function() {
    .require_training_data(model, c("x", "y"))
    ppforest2_oob_error(model, model$x, model$y)
  })
}


#' Out-of-bag predictions for a random forest.
#'
#' Returns predictions for each training row using only trees that did
#' not see that row in their bootstrap sample. Observations with no OOB
#' tree are represented as `NA` in both modes: `NA` at the factor level
#' for classification, and `NA_real_` for regression. Filter with the
#' standard `is.na()` idiom.
#'
#' @param model A \code{pprf} forest model.
#' @return A factor (classification) or numeric vector (regression), length `n`.
#' @seealso \code{\link{oob_error}}, \code{\link{oob_samples}}
#' @export
oob_predictions <- function(model) UseMethod("oob_predictions")

#' @export
oob_predictions.default <- function(model) {
  stop(
    "`oob_predictions()` is only defined for `pprf` forest models, not objects of class '",
    paste(class(model), collapse = "/"), "'.",
    call. = FALSE
  )
}

#' @export
oob_predictions.pprf_classification <- function(model) {
  .cached_or_compute(model, "oob_predictions", function() {
    .require_training_data(model, "x")
    # Sentinel: C++ `oob_predict` emits `NaN` for rows that had no OOB
    # tree, in both modes. The Rcpp wrapper's `to_r_indices` shift adds
    # +1 to every element to convert 0-based C++ group ids to 1-based
    # R factor indices — `NaN + 1` is still `NaN`, so the sentinel
    # survives intact. We then remap `NaN → NA_real_` explicitly (the
    # same step the regression accessor does) so the downstream
    # `as.integer` coercion doesn't emit "NAs introduced by coercion"
    # warnings, and the factor records each no-OOB row as a missing
    # level cleanly.
    raw <- ppforest2_oob_predict(model, model$x)
    raw[is.nan(raw)] <- NA_real_
    factor(model$groups[as.integer(raw)], levels = model$groups)
  })
}

#' @export
oob_predictions.pprf_regression <- function(model) {
  .cached_or_compute(model, "oob_predictions", function() {
    .require_training_data(model, "x")
    # C++ emits NaN for rows with no OOB tree (the natural float sentinel,
    # used identically in classification — see `oob_predictions.pprf_classification`).
    # Remap to NA_real_ here so callers can use the plain `is.na()` idiom
    # without having to remember the underlying NaN.
    out <- as.numeric(ppforest2_oob_predict(model, model$x))
    out[is.nan(out)] <- NA_real_
    out
  })
}


#' Out-of-bag row indices per tree.
#'
#' Returns a list where element `i` is the integer vector of row indices
#' (1-based) that were **not** in the bootstrap sample of tree `i`.
#'
#' @param model A \code{pprf} forest model.
#' @return A list of integer vectors, one per tree.
#' @seealso \code{\link{bag_samples}}, \code{\link{oob_error}}
#' @export
oob_samples <- function(model) UseMethod("oob_samples")

#' @export
oob_samples.default <- function(model) {
  stop(
    "`oob_samples()` is only defined for `pprf` forest models, not objects of class '",
    paste(class(model), collapse = "/"), "'.",
    call. = FALSE
  )
}

#' @export
oob_samples.pprf <- function(model) {
  .require_training_data(model, "x")
  n <- nrow(model$x)
  lapply(model$trees, function(t) setdiff(seq_len(n), t$sample_indices + 1L))
}


#' In-bag row indices per tree.
#'
#' Returns a list where element `i` is the integer vector of row indices
#' (1-based, with replacement) drawn into the bootstrap sample of tree `i`.
#'
#' @param model A \code{pprf} forest model.
#' @return A list of integer vectors, one per tree.
#' @seealso \code{\link{oob_samples}}
#' @export
bag_samples <- function(model) UseMethod("bag_samples")

#' @export
bag_samples.default <- function(model) {
  stop(
    "`bag_samples()` is only defined for `pprf` forest models, not objects of class '",
    paste(class(model), collapse = "/"), "'.",
    call. = FALSE
  )
}

#' @export
bag_samples.pprf <- function(model) {
  lapply(model$trees, function(t) t$sample_indices + 1L)
}


# ---------------------------------------------------------------------------
# Variable-importance accessors (lazy for OOB-requiring variants).
# ---------------------------------------------------------------------------

#' Permuted variable importance for a random forest.
#'
#' For each feature, measures the drop in OOB accuracy (classification) or
#' the increase in normalised MSE (regression) after randomly permuting
#' that feature across the OOB observations. Computed lazily from the
#' training data stored on the model; the result is cached.
#'
#' **Sign semantics.** Entries may be **negative**. That is not an error
#' and not a sentinel: it means permuting the feature did not degrade OOB
#' fit on average — the feature's signal sits at or below the noise floor
#' of the permutation procedure. Interpret negative or near-zero entries
#' as "no evidence of importance"; rely on the ranking rather than
#' clipping at zero or normalizing. The scale is already comparable
#' within a fitted model.
#'
#' @param model A \code{pprf} forest model.
#' @return A numeric vector, one entry per feature. Negative values are
#'   meaningful (see Sign semantics above).
#' @export
permuted_importance <- function(model) UseMethod("permuted_importance")

#' @export
permuted_importance.default <- function(model) {
  stop(
    "`permuted_importance()` is only defined for `pprf` forest models.",
    call. = FALSE
  )
}

#' @export
permuted_importance.pprf <- function(model) {
  # `model$y` is mode-correct for both classification (1-based factor codes,
  # which the C++ binding shifts to 0-based when needed) and regression
  # (the continuous response). Previously the regression branch passed a
  # 0/1 median-split indicator here, making MSE/NMSE meaningless — fixed
  # by unifying `model$y` with the continuous response in `validate_data()`.
  .cached_or_compute(model, "permuted_importance", function() {
    .require_training_data(model, c("x", "y"))
    ppforest2_vi_permuted_forest(model, model$x, model$y, model$seed)
  })
}


#' Weighted projection variable importance for a random forest.
#'
#' Weights each tree's projection-based importance by a per-tree OOB
#' quality score — `1 - error_rate` in `[0, 1]` for classification, and
#' `max(0, 1 - NMSE)` in `[0, 1]` for regression — then aggregates
#' `I_s × |a_j|` over splits. Computed lazily from the training data
#' stored on the model; the result is cached.
#'
#' **Sign semantics.** Entries are non-negative by construction (weights
#' and per-split contributions are both non-negative). A zero entry means
#' "this feature never appeared in a weighted OOB-contributing split,"
#' not "within noise." Contrast with \code{\link{permuted_importance}},
#' where negative values are meaningful. Do not re-normalize — rely on
#' the ranking.
#'
#' @param model A \code{pprf} forest model.
#' @return A non-negative numeric vector, one entry per feature.
#' @export
weighted_importance <- function(model) UseMethod("weighted_importance")

#' @export
weighted_importance.default <- function(model) {
  stop(
    "`weighted_importance()` is only defined for `pprf` forest models.",
    call. = FALSE
  )
}

#' @export
weighted_importance.pprf <- function(model) {
  # See the note on `permuted_importance.pprf` — `model$y` is now
  # mode-correct for both classification and regression.
  .cached_or_compute(model, "weighted_importance", function() {
    .require_training_data(model, c("x", "y"))
    ppforest2_vi_weighted_forest(model, model$x, model$y, model$vi$scale)
  })
}


# ---------------------------------------------------------------------------
# Shared S3 methods at the ppmodel level (both pptr and pprf).
# ---------------------------------------------------------------------------

#' Formula extractor for ppforest2 models.
#'
#' @param x A \code{pptr} or \code{pprf} model.
#' @param ... Unused.
#' @return The formula the model was trained with, or `NULL` for matrix-interface fits.
#' @export
formula.ppmodel <- function(x, ...) {
  x$formula
}


#' Number of observations used to fit a ppforest2 model.
#'
#' Implements the standard \code{stats::nobs()} contract so downstream tools
#' (\code{step()}, broom's \code{glance()}, information criteria) can ask for
#' the training-sample size.
#'
#' @param object A \code{pptr} or \code{pprf} model.
#' @param ... Unused.
#' @return Integer scalar. Returns \code{NA_integer_} for models loaded from
#'   JSON without their original training data.
#' @export
nobs.ppmodel <- function(object, ...) {
  if (is.null(object$x)) return(NA_integer_)
  nrow(object$x)
}


#' Fitted (in-sample) predictions from a ppforest2 model.
#'
#' Returns predictions on the training data — a factor for classification,
#' a numeric vector for regression. For forests, training predictions are
#' optimistic; use \code{\link{oob_predictions}()} for an honest estimate.
#'
#' @param object A \code{pptr} or \code{pprf} model.
#' @param ... Unused.
#' @return A factor (classification) or numeric vector (regression), length
#'   equal to the number of training observations.
#' @seealso \code{\link{residuals.ppmodel}}, \code{\link{oob_predictions}}
#' @export
fitted.ppmodel <- function(object, ...) {
  .require_training_data(object, "x")
  # Dispatches through the model's own predict method, which is mode- and
  # class-aware (pprf vs pptr × classification vs regression). For
  # classification this returns a factor; for regression a numeric vector.
  predict(object, object$x)
}


#' Residuals from a regression ppforest2 model.
#'
#' Returns \code{y - fitted(model)}. Only defined for regression models —
#' classification residuals have no canonical scalar form, so this method
#' errors on classification models rather than inventing a convention.
#'
#' @param object A \code{pprf} or \code{pptr} regression model.
#' @param ... Unused.
#' @return A numeric vector of length \code{nobs(object)}.
#' @seealso \code{\link{fitted.ppmodel}}
#' @export
residuals.ppmodel <- function(object, ...) {
  if (!identical(object$mode, "regression")) {
    stop(
      "`residuals()` is only defined for regression models. ",
      "For classification, compare `fitted(model)` against the response directly, ",
      "or use `oob_predictions(model)` for an out-of-bag misclassification view.",
      call. = FALSE
    )
  }
  .require_training_data(object, c("x", "y"))
  as.numeric(object$y) - as.numeric(fitted(object))
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Throws a clear error if required training-data fields are missing on the
# model (e.g. the model was loaded from JSON without the original x/y).
# Distinguishes between "loaded model without data" (JSON doesn't carry x/y,
# user should reattach) and "metrics were never computed at save time" (the
# saved JSON had `include_metrics = FALSE`, so priming the cache was a
# no-op and no recomputation is possible).
.require_training_data <- function(model, fields) {
  for (f in fields) {
    if (is.null(model[[f]])) {
      # Heuristic: if the model has a `.cache` environment and `training_spec`
      # populated, it was almost certainly loaded from JSON. Mention both
      # reattachment and the save-time-metrics case so the user can pick.
      loaded_from_json <- !is.null(model$training_spec) && !is.null(model$.cache)
      if (loaded_from_json) {
        stop(
          "Required field `", f, "` is not available on the loaded model. ",
          "Either (a) re-attach the original training data to `model$", f,
          "` before calling this accessor, or (b) re-save the source model ",
          "with `save_json(model, path, include_metrics = TRUE)` so the ",
          "relevant OOB / variable-importance values are preserved in the JSON.",
          call. = FALSE
        )
      }
      stop(
        "Required field `", f, "` is not available on the model -- ",
        "this model was probably loaded from JSON without the original training data.",
        call. = FALSE
      )
    }
  }
  invisible(NULL)
}
