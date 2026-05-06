#' Label-based grouping strategy.
#'
#' Creates a grouping strategy that routes all observations of a group to the
#' same child node. This is the default grouping strategy for classification.
#'
#' @return A \code{grouping_strategy} object.
#' @seealso \code{\link{pptr}}, \code{\link{pprf}}
#' @examples
#' grouping_by_label()
#'
#' @export
grouping_by_label <- function() {
  structure(list(name = "by_label", display_name = "By label"), class = "grouping_strategy")
}

#' Cutpoint-based grouping strategy (regression).
#'
#' Creates a grouping strategy for regression trees: observations are split by
#' the cutpoint in projected space, then each child's observations are sorted
#' by the continuous response and median-split into 2 new groups for the next
#' projection-pursuit step. This is the default grouping strategy for regression.
#'
#' @return A \code{grouping_strategy} object.
#' @seealso \code{\link{pptr}}, \code{\link{pprf}}, \code{\link{grouping_by_label}}
#' @examples
#' grouping_by_cutpoint()
#'
#' @export
grouping_by_cutpoint <- function() {
  structure(list(name = "by_cutpoint", display_name = "By cutpoint"), class = "grouping_strategy")
}
