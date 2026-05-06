#' Largest-gap binarization strategy.
#'
#' Creates a binarization strategy that reduces multiclass nodes to binary
#' by projecting group means and splitting at the largest gap. Default
#' for classification specs.
#'
#' @return A \code{binarize_strategy} object.
#' @seealso \code{\link{pptr}}, \code{\link{pprf}}, \code{\link{binarize_disabled}}
#' @examples
#' binarize_largest_gap()
#'
#' @export
binarize_largest_gap <- function() {
  structure(list(name = "largest_gap", display_name = "Largest gap"), class = "binarize_strategy")
}

#' Disabled binarization strategy (placeholder).
#'
#' Placeholder binarizer for specs where binarization never fires —
#' notably regression, where `grouping_by_cutpoint()` always produces
#' a 2-group partition at each node. Selecting \code{binarize_disabled()}
#' documents that intent explicitly; if binarization is ever invoked at
#' runtime with this strategy configured, training aborts with a clear
#' error rather than silently passing through. Default for regression
#' specs.
#'
#' @return A \code{binarize_strategy} object.
#' @seealso \code{\link{binarize_largest_gap}}
#' @examples
#' binarize_disabled()
#'
#' @export
binarize_disabled <- function() {
  structure(list(name = "disabled", display_name = "Disabled"), class = "binarize_strategy")
}
