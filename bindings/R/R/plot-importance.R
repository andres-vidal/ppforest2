# ======================================================================
# Variable importance plot
# ======================================================================

#' Display labels for each variable importance metric.
#'
#' Keys must match names in \code{model$vi} (set during training/forest
#' construction).
#' @noRd
metric_labels <- c(
  projections = "Projection",
  weighted    = "Weighted",
  permuted    = "Permuted"
)

#' Render a horizontal bar chart of variable importance.
#'
#' Supports multiple importance metrics (projection, weighted, permuted).
#' When \code{metric} is \code{NULL} and multiple metrics are available,
#' bars are colour-coded and dodged.  When a single metric is selected
#' (or only one is available), bars use the neutral
#' \code{ppforest2_col_bar} colour.  Variables are ordered by the
#' plotted metric's values (descending).
#'
#' @param model A pptr or pprf model with \code{$vi} (list of named
#'   numeric vectors) and \code{$x} (for variable names).
#' @param metric Character string selecting a single importance metric
#'   to plot: \code{"projections"}, \code{"weighted"}, or
#'   \code{"permuted"}.  \code{NULL} (default) shows all available
#'   metrics together.
#' @param ... Currently unused.
#' @return A ggplot2 object.
#' @noRd
plot_importance <- function(model, metric = NULL, ...) {
  vnames <- get_variable_names(model)

  # For forests, weighted and permuted are lazy accessors; for trees only
  # projections is available. Build the set of available metrics per the
  # model class, not field presence on `$vi`.
  is_forest <- inherits(model, "pprf")
  available <- if (is_forest) c("projections", "weighted", "permuted") else "projections"

  get_values <- function(m) {
    switch(m,
      projections = model$vi$projections,
      weighted    = weighted_importance(model),
      permuted    = permuted_importance(model)
    )
  }

  if (!is.null(metric)) {
    metric <- match.arg(metric, names(metric_labels))
    if (!(metric %in% available)) {
      stop(
        "Metric '", metric, "' is not available for this model. ",
        "Available: ", paste(available, collapse = ", "),
        call. = FALSE
      )
    }
    available <- metric
  }

  dfs <- list()
  for (m in available) {
    values <- get_values(m)
    dfs[[m]] <- data.frame(
      variable   = vnames,
      importance = values,
      metric     = unname(metric_labels[m]),
      stringsAsFactors = FALSE
    )
  }

  df <- do.call(rbind, dfs)

  # Order variables by first (or only) metric's values
  first_values <- get_values(available[1])
  df$variable <- factor(df$variable, levels = vnames[order(first_values)])
  df$metric <- factor(df$metric, levels = unname(metric_labels[available]))

  if (length(available) > 1) {
    p <- ggplot2::ggplot(df, ggplot2::aes(x = importance, y = variable, fill = metric)) +
      ggplot2::geom_col(position = "dodge") +
      ggplot2::labs(
        title = "Variable Importance",
        x     = "Importance",
        y     = NULL,
        fill  = "Metric"
      ) +
      ppforest2_theme() +
      ggplot2::theme(panel.grid.major.y = ggplot2::element_blank())
  } else {
    title <- paste0("Variable Importance (", metric_labels[available], ")")
    p <- ggplot2::ggplot(df, ggplot2::aes(x = importance, y = variable)) +
      ggplot2::geom_col(fill = ppforest2_col_bar()) +
      ggplot2::labs(
        title = title,
        x     = "Importance",
        y     = NULL
      ) +
      ppforest2_theme() +
      ggplot2::theme(panel.grid.major.y = ggplot2::element_blank())
  }

  p
}

#' Render a grid of individual importance plots, one per metric.
#'
#' Used by \code{plot.pprf()} to show all importance metrics side by side,
#' each ordered independently.
#'
#' @param model A pprf model.
#' @param ... Currently unused.
#' @return A patchwork object (ggplot-compatible, works with ggsave).
#' @noRd
plot_importance_grid <- function(model, ...) {
  check_patchwork()
  available <- intersect(names(metric_labels), names(model$vi))

  plots <- lapply(available, function(m) {
    # Use short subtitle per panel instead of full title
    plot_importance(model, metric = m) +
      ggplot2::labs(title = metric_labels[m]) +
      ggplot2::theme(plot.margin = ggplot2::margin(5, 10, 5, 10))
  })

  Reduce(`+`, plots) +
    patchwork::plot_layout(nrow = 1) +
    patchwork::plot_annotation(title = "Variable Importance")
}
