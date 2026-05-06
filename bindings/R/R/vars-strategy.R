#' Uniform random variable selection strategy.
#'
#' Creates a variable selection strategy that randomly selects a subset
#' of variables at each split. Used with \code{\link{pprf}} for random forests.
#'
#' Exactly one of \code{n_vars} or \code{p_vars} may be specified. When
#' \code{p_vars} is used, it is stored as-is and resolved to an integer count
#' later by \code{\link{pprf}} once the number of features is known.
#'
#' @param n_vars The number of variables to consider at each split (integer).
#'   Cannot be used together with \code{p_vars}.
#' @param p_vars The proportion of variables to consider at each split
#'   (number between 0 and 1, exclusive). Resolved to an integer count when
#'   the number of features is known. Cannot be used together with \code{n_vars}.
#' @return A \code{vars_strategy} object.
#' @seealso \code{\link{pprf}}, \code{\link{vars_all}}, \code{\link{pp_pda}}, \code{\link{cutpoint_mean_of_means}}
#' @examples
#' # Select 2 variables at each split
#' vars_uniform(n_vars = 2)
#'
#' # Select half the variables at each split
#' vars_uniform(p_vars = 0.5)
#'
#' # Use with pprf
#' pprf(Species ~ ., data = iris, vars = vars_uniform(n_vars = 2))
#'
#' @export
vars_uniform <- function(n_vars = NULL, p_vars = NULL) {
  if (!is.null(n_vars) && !is.null(p_vars))
    stop("Only one of `n_vars` or `p_vars` may be specified, not both.")

  if (!is.null(n_vars) && (!is.numeric(n_vars) || length(n_vars) != 1 || n_vars < 1 || n_vars != as.integer(n_vars)))
    stop("`n_vars` must be a positive integer greater than 0.")

  if (!is.null(p_vars) && (!is.numeric(p_vars) || length(p_vars) != 1 || p_vars <= 0 || p_vars > 1))
    stop("`p_vars` must be a single number between 0 (exclusive) and 1 (inclusive).")

  structure(list(name = "uniform", display_name = "Uniform random", count = n_vars, p_vars = p_vars), class = "vars_strategy")
}

#' All-variables selection strategy.
#'
#' Creates a variable selection strategy that uses all variables at each
#' split. This is the default for single trees (\code{\link{pptr}}).
#'
#' @return A \code{vars_strategy} object.
#' @seealso \code{\link{pptr}}, \code{\link{vars_uniform}}, \code{\link{pp_pda}}, \code{\link{cutpoint_mean_of_means}}
#' @examples
#' vars_all()
#'
#' @export
vars_all <- function() {
  structure(list(name = "all", display_name = "All variables"), class = "vars_strategy")
}
