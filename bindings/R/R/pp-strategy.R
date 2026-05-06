#' @useDynLib ppforest2
#' @importFrom Rcpp evalCpp
NULL

#' PDA projection pursuit strategy.
#'
#' Creates a Penalized Discriminant Analysis (PDA) projection pursuit strategy
#' for use with \code{\link{pptr}} or \code{\link{pprf}}.
#'
#' @param lambda A regularization parameter between 0 and 1.
#'   If \code{lambda = 0}, the model uses Linear Discriminant Analysis (LDA).
#'   If \code{lambda > 0}, the model uses Penalized Discriminant Analysis (PDA).
#' @return A \code{pp_strategy} object.
#' @seealso \code{\link{pptr}}, \code{\link{pprf}}, \code{\link{vars_uniform}}, \code{\link{vars_all}}, \code{\link{cutpoint_mean_of_means}}
#' @examples
#' # PDA with lambda = 0.5
#' pp_pda(0.5)
#'
#' # Use with pptr
#' pptr(Species ~ ., data = iris, pp = pp_pda(0.5))
#'
#' # Use with pprf
#' pprf(Species ~ ., data = iris, pp = pp_pda(0.5), vars = vars_uniform(n_vars = 2))
#'
#' @export
pp_pda <- function(lambda = 0) {
  if (!is.numeric(lambda) || length(lambda) != 1 || lambda < 0 || lambda > 1)
    stop("`lambda` must be a single number between 0 and 1.")
    
  structure(list(name = "pda", display_name = ifelse(lambda == 0, "LDA", "PDA"), lambda = lambda), class = "pp_strategy")
}
