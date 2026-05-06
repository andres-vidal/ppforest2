#' Majority-vote leaf strategy.
#'
#' Creates a leaf strategy that assigns the majority group label as the
#' leaf prediction. When groups are tied, the smallest label wins.
#' This is the default leaf strategy for classification.
#'
#' @return A \code{leaf_strategy} object.
#' @seealso \code{\link{pptr}}, \code{\link{pprf}}
#' @examples
#' leaf_majority_vote()
#'
#' @export
leaf_majority_vote <- function() {
  structure(list(name = "majority_vote", display_name = "Majority vote"), class = "leaf_strategy")
}

#' Mean-response leaf strategy.
#'
#' Creates a leaf strategy that predicts the mean of the continuous response
#' values for the observations in the leaf. Used for regression trees; requires
#' a numeric response.
#'
#' @return A \code{leaf_strategy} object.
#' @seealso \code{\link{pptr}}, \code{\link{pprf}}, \code{\link{leaf_majority_vote}}
#' @examples
#' leaf_mean_response()
#'
#' @export
leaf_mean_response <- function() {
  structure(list(name = "mean_response", display_name = "Mean response"), class = "leaf_strategy")
}
