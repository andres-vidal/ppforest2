#' Pure-node stopping rule.
#'
#' Creates a stopping rule that stops splitting when a node contains only one
#' group. This is the default stopping rule for classification.
#'
#' @return A \code{stop_strategy} object.
#' @seealso \code{\link{pptr}}, \code{\link{pprf}}
#' @examples
#' stop_pure_node()
#'
#' @export
stop_pure_node <- function() {
  structure(list(name = "pure_node", display_name = "Pure node"), class = "stop_strategy")
}

#' Minimum-size stopping rule.
#'
#' Creates a stopping rule that stops splitting when a node has fewer than
#' \code{min_size} observations. Used primarily for regression trees.
#'
#' @param min_size Minimum node size to allow a split (default: 5).
#' @return A \code{stop_strategy} object.
#' @seealso \code{\link{stop_min_variance}}, \code{\link{stop_any}}
#' @examples
#' stop_min_size(5)
#'
#' @export
stop_min_size <- function(min_size = 5L) {
  min_size <- as.integer(min_size)
  if (min_size < 2L) {
    stop("`min_size` must be an integer >= 2 (a node with 0 or 1 observations is unsplittable).")
  }
  structure(
    list(name = "min_size", min_size = min_size, display_name = paste0("Min size (", min_size, ")")),
    class = "stop_strategy"
  )
}

#' Minimum-variance stopping rule.
#'
#' Creates a stopping rule that stops splitting when the within-node response
#' variance falls below \code{threshold}. Used primarily for regression trees;
#' requires a continuous response.
#'
#' @param threshold Variance threshold below which to stop splitting (default: 0.01).
#' @return A \code{stop_strategy} object.
#' @seealso \code{\link{stop_min_size}}, \code{\link{stop_any}}
#' @examples
#' stop_min_variance(0.01)
#'
#' @export
stop_min_variance <- function(threshold = 0.01) {
  threshold <- as.numeric(threshold)
  if (threshold < 0) {
    stop("`threshold` must be non-negative.")
  }
  # Matches `MinVariance::display_name` on the C++ side, which uses
  # `defaultfloat` with precision 6 so small thresholds like 1e-6 render
  # as "1e-06" instead of rounding to "0.0000".
  structure(
    list(name = "min_variance", threshold = threshold, display_name = sprintf("Min variance (%s)", formatC(threshold, format = "g", digits = 6))),
    class = "stop_strategy"
  )
}

#' Maximum-depth stopping rule.
#'
#' Creates a stopping rule that stops splitting when a node's depth reaches
#' \code{max_depth}. Depth is zero-based at the root, so \code{max_depth(k)}
#' allows at most \code{k + 1} levels. Mode-agnostic: useful for bounding
#' tree complexity in both classification and regression trees.
#'
#' @param max_depth Maximum depth (non-negative integer; 0 produces a
#'   root-only stump).
#' @return A \code{stop_strategy} object.
#' @seealso \code{\link{stop_min_size}}, \code{\link{stop_any}}
#' @examples
#' stop_max_depth(5)
#'
#' @export
stop_max_depth <- function(max_depth) {
  max_depth <- as.integer(max_depth)
  if (is.na(max_depth) || max_depth < 0L) {
    stop("`max_depth` must be a non-negative integer.")
  }
  structure(
    list(name = "max_depth", max_depth = max_depth, display_name = paste0("Max depth (", max_depth, ")")),
    class = "stop_strategy"
  )
}

#' Composite stopping rule (logical OR).
#'
#' Creates a composite stopping rule that fires when any of the child rules
#' fires. Useful for combining multiple criteria, e.g. \code{stop_any(stop_min_size(5),
#' stop_min_variance(0.01))} for regression.
#'
#' @param ... Two or more \code{stop_strategy} objects to combine.
#' @return A \code{stop_strategy} object.
#' @seealso \code{\link{stop_min_size}}, \code{\link{stop_min_variance}}, \code{\link{stop_pure_node}}
#' @examples
#' stop_any(stop_min_size(5), stop_min_variance(0.01))
#'
#' @export
stop_any <- function(...) {
  rules <- list(...)
  if (length(rules) < 1L) {
    stop("`stop_any` requires at least one stop rule.")
  }
  for (r in rules) {
    if (!inherits(r, "stop_strategy")) {
      stop("All arguments to `stop_any` must be stop_strategy objects.")
    }
  }
  names <- vapply(rules, function(r) r$display_name %||% r$name, character(1))
  structure(
    list(name = "any", rules = rules, display_name = paste0("Any(", paste(names, collapse = ", "), ")")),
    class = "stop_strategy"
  )
}
