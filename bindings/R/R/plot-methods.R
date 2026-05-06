# ======================================================================
# S3 plot methods
# ======================================================================

#' Plot a pptr model.
#'
#' Visualizes a pptr model. By default, shows a mosaic overview with tree
#' structure, decision boundaries, and variable importance. Use \code{type}
#' to show individual plots.
#'
#' @param x A pptr model.
#' @param type Character string specifying the plot type. \code{NULL} (default)
#'   shows a mosaic overview. Other options: \code{"structure"} for tree with
#'   embedded histograms, \code{"importance"} for variable importance,
#'   \code{"projection"} for projected data at a node,
#'   \code{"boundaries"} for decision boundaries in feature space.
#' @param metric Character string selecting a single importance metric
#'   to plot: \code{"projections"}, \code{"weighted"}, or
#'   \code{"permuted"} (availability depends on the model).  \code{NULL}
#'   (default) shows all available metrics together.  Only used when
#'   \code{type = "importance"}.
#' @param node Integer index of the node for projection plots (1-based, breadth-first
#'   order). Defaults to 1 (root node). Only used when \code{type = "projection"}.
#' @param ... Additional arguments passed to the internal plotting function.
#' @return A ggplot2-compatible object (invisibly). The mosaic layout returns
#'   a patchwork object that works with \code{ggplot2::ggsave()}.
#' @examples
#' \dontrun{
#' model <- pptr(Species ~ ., data = iris)
#' plot(model)                         # mosaic overview
#' plot(model, type = "structure")     # tree structure only
#' plot(model, type = "importance")    # variable importance
#' plot(model, type = "projection")   # projection histogram
#' plot(model, type = "boundaries")   # decision boundaries
#' }
#' @export
plot.pptr <- function(x, type = NULL, metric = NULL, node = 1L, ...) {
  check_ggplot2()

  if (is.null(type)) {
    p <- plot_mosaic(x, ...)
    print(p)
    return(invisible(p))
  }

  type <- match.arg(type, c("structure", "importance", "projection", "boundaries"))

  p <- switch(type,
    structure  = plot_tree_structure(x, ...),
    importance = plot_importance(x, metric = metric, ...),
    projection = plot_projection(x, node = node, ...),
    boundaries = plot_boundaries(x, ...)
  )

  if (inherits(p, "gtable")) {
    grid::grid.newpage()
    grid::grid.draw(p)
  } else {
    print(p)
  }
  invisible(p)
}

#' Plot a pprf model.
#'
#' Visualizes a pprf model. By default, shows variable importance with
#' one plot per metric side by side. Use \code{metric} to show a single
#' importance metric.
#'
#' @param x A pprf model.
#' @param type Character string specifying the plot type.
#'   \code{"importance"} (default) shows variable importance,
#'   \code{"structure"} shows a specific tree with embedded histograms,
#'   \code{"projection"} shows projected data at a node,
#'   \code{"boundaries"} shows decision boundaries of a specific tree.
#' @param metric Character string selecting a single importance metric
#'   to plot: \code{"projections"}, \code{"weighted"}, or
#'   \code{"permuted"}.  \code{NULL} (default) shows all available
#'   metrics side by side in separate panels.  Only used when
#'   \code{type = "importance"}.
#' @param tree_index Integer index of the tree to plot (1-based). Only used when
#'   \code{type = "structure"}, \code{type = "projection"}, or
#'   \code{type = "boundaries"}. Defaults to 1.
#' @param node Integer index of the node for projection plots. Defaults to 1 (root).
#'   Only used when \code{type = "projection"}.
#' @param ... Additional arguments passed to the internal plotting function.
#' @return A ggplot2-compatible object (invisibly). The importance grid returns
#'   a patchwork object that works with \code{ggplot2::ggsave()}.
#' @examples
#' \dontrun{
#' forest <- pprf(Species ~ ., data = iris, size = 10)
#' plot(forest)                                    # all metrics side by side
#' plot(forest, metric = "permuted")               # single metric
#' plot(forest, type = "structure", tree_index = 1)
#' plot(forest, type = "projection", tree_index = 1)
#' }
#' @export
plot.pprf <- function(x, type = "importance", metric = NULL,
                          tree_index = 1L, node = 1L, ...) {
  check_ggplot2()
  type <- match.arg(type, c("importance", "structure", "projection", "boundaries"))

  if (type %in% c("structure", "projection", "boundaries")) {
    tree <- x$trees[[tree_index]]
    tree$x <- x$x
    tree$y <- x$y
    tree$groups <- x$groups
  }

  if (type == "importance" && is.null(metric)) {
    p <- plot_importance_grid(x, ...)
    print(p)
    return(invisible(p))
  }

  p <- switch(type,
    importance = plot_importance(x, metric = metric, ...),
    structure  = plot_tree_structure(tree, ...),
    projection = plot_projection(tree, node = node, ...),
    boundaries = plot_boundaries(tree, ...)
  )

  if (inherits(p, "gtable")) {
    grid::grid.newpage()
    grid::grid.draw(p)
  } else {
    print(p)
  }
  invisible(p)
}
