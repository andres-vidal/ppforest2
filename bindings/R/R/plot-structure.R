# ======================================================================
# Tree structure plot
# ======================================================================

#' Compute histogram bar rectangles for one internal (condition) node.
#'
#' Bins the projected values by group, then maps bin coordinates from
#' projection space to the node's rectangle in the tree layout canvas.
#' This is NOT a standard ggplot2 \code{geom_histogram} — it produces raw
#' rectangle coordinates (xmin/xmax/ymin/ymax) that are drawn with
#' \code{geom_rect} so each node's histogram is embedded inside its box.
#'
#' @param nd A node data list from \code{ppforest2_tree_node_data()},
#'   with \code{$projected} (numeric), \code{$groups} (integer),
#'   and \code{$cutpoint} (numeric).
#' @param cx,cy Center position of the node on the layout canvas.
#' @param node_w,node_h Width and height of the node rectangle.
#' @param group_labels Character vector of group names (model$groups).
#' @param n_bins Number of histogram bins (default 15).
#' @return List with:
#'   \describe{
#'     \item{bars}{data.frame with xmin, xmax, ymin, ymax, group columns}
#'     \item{thr_x}{x-coordinate of the cutpoint line in canvas space}
#'     \item{thr_y_min, thr_y_max}{y-extent of the cutpoint line}
#'     \item{ticks}{data.frame with x, y, label for axis tick labels}
#'   }
#'   or NULL if no non-empty bins exist.
#' @noRd
compute_histogram_bars <- function(nd, cx, cy, node_w, node_h, group_labels, n_bins = 15) {
  vals <- nd$projected
  cls <- group_labels[nd$groups]
  cutpoint <- nd$cutpoint

  breaks <- seq(min(vals), max(vals), length.out = n_bins + 1)
  unique_groups <- unique(cls)

  bars <- list()
  for (cl in unique_groups) {
    cl_vals <- vals[cls == cl]
    h <- graphics::hist(cl_vals, breaks = breaks, plot = FALSE)
    if (any(h$counts > 0)) {
      bars[[length(bars) + 1L]] <- data.frame(
        bin_left  = h$breaks[-length(h$breaks)],
        bin_right = h$breaks[-1],
        count     = h$counts,
        group     = cl,
        stringsAsFactors = FALSE
      )
    }
  }

  if (length(bars) == 0) return(NULL)

  bar_df <- do.call(rbind, bars)
  bar_df <- bar_df[bar_df$count > 0, , drop = FALSE]

  if (nrow(bar_df) == 0) return(NULL)

  # Map projected-value-space to node coordinate space
  val_min <- min(breaks)
  val_max <- max(breaks)
  val_range <- val_max - val_min
  if (val_range == 0) val_range <- 1

  margin <- 0.05
  draw_w <- node_w * (1 - 2 * margin)
  draw_h <- node_h * (1 - 2 * margin)

  max_count <- max(bar_df$count)

  bar_df$xmin <- cx - node_w / 2 + node_w * margin + (bar_df$bin_left - val_min) / val_range * draw_w
  bar_df$xmax <- cx - node_w / 2 + node_w * margin + (bar_df$bin_right - val_min) / val_range * draw_w
  bar_df$ymin <- cy - node_h / 2 + node_h * margin
  bar_df$ymax <- bar_df$ymin + bar_df$count / max_count * draw_h

  # Threshold line in data coordinates
  thr_x <- cx - node_w / 2 + node_w * margin + (cutpoint - val_min) / val_range * draw_w

  # Axis tick data
  tick_y <- cy - node_h / 2
  tick_labels <- data.frame(
    x = c(cx - node_w / 2 + node_w * margin, cx + node_w / 2 - node_w * margin),
    y = tick_y,
    label = formatC(c(val_min, val_max), format = "g", digits = 3),
    stringsAsFactors = FALSE
  )

  list(bars = bar_df, thr_x = thr_x, thr_y_min = cy - node_h / 2 + node_h * margin, thr_y_max = cy + node_h / 2 - node_h * margin, ticks = tick_labels)
}

#' Format a projector vector as a compact equation string.
#'
#' Shows the top-k coefficients by absolute magnitude as a sum like
#' \code{"0.71\u00b7SL \u2212 0.29\u00b7PW"}.  If terms were omitted,
#' appends \code{"+ \u2026"}.
#'
#' @param projector Numeric vector of projector coefficients.
#' @param var_names Character vector of variable names (same length as projector).
#' @param max_terms Maximum number of terms to display.
#' @return A single character string.
#' @noRd
format_projector <- function(projector, var_names, max_terms = 4L) {
  ord <- order(abs(projector), decreasing = TRUE)
  n_nonzero <- sum(projector != 0)
  n_show <- min(max_terms, n_nonzero)

  if (n_show == 0) return("0")

  top_idx <- ord[seq_len(n_show)]
  coefs <- projector[top_idx]
  names_top <- var_names[top_idx]

  parts <- character(n_show)
  for (i in seq_len(n_show)) {
    # Significant digits rather than fixed decimals.  The projector is normalized
    # so the values it projects always have the same spread whatever the units of
    # the input, which leaves its coefficients at a magnitude set by the data
    # rather than by the split.  Same precision as the axis tick labels.
    coef_str <- formatC(abs(coefs[i]), format = "g", digits = 3)
    # Drop leading zero: "0.71" -> ".71"
    coef_str <- sub("^0\\.", ".", coef_str)
    term <- paste0(coef_str, "\u00b7", names_top[i])

    if (i == 1L) {
      parts[i] <- if (coefs[i] < 0) paste0("\u2212", term) else term
    } else {
      parts[i] <- if (coefs[i] < 0) paste0(" \u2212 ", term) else paste0(" + ", term)
    }
  }

  result <- paste0(parts, collapse = "")

  if (n_nonzero > n_show) {
    result <- paste0(result, " + \u2026")
  }

  result
}

#' Render the tree structure as a ggplot2 plot.
#'
#' Draws a tree diagram on a 2D canvas with:
#' \itemize{
#'   \item Edges connecting parent to child nodes (grey segments with labels)
#'   \item Internal nodes as white rectangles containing stacked group
#'     histograms of the projected values at that split, with a dashed
#'     cutpoint line
#'   \item A compact projector equation below each internal node showing
#'     the top coefficients (controlled by \code{max_terms})
#'   \item Leaf nodes as group-colored rectangles with the predicted group
#'     label
#' }
#'
#' Layout positions come from the C++ \code{compute_tree_layout()} (pre-order,
#' left-aligned).  Per-node projection data comes from
#' \code{ppforest2_tree_node_data()} which routes all training observations
#' through the tree.
#'
#' @param model A pptr model with \code{$root}, \code{$x}, \code{$y},
#'   \code{$groups}.
#' @param max_terms Maximum number of projector terms to display per node
#'   (default 3).  Only the largest-magnitude coefficients are shown.
#' @param ... Currently unused.
#' @return A ggplot2 object (theme_void, no axes).
#' @noRd
plot_tree_structure <- function(model, max_terms = 3L, ...) {
  nodes <- ppforest2_tree_node_data(model, model$x, model$y)
  group_labels <- model$groups
  group_colors <- get_group_colors(group_labels)
  var_names <- get_variable_names(model)

  node_w <- ppforest2_node_w()
  node_h <- ppforest2_node_h()
  leaf_w <- ppforest2_leaf_w()
  leaf_h <- ppforest2_leaf_h()

  layout <- ppforest2_tree_layout(model)
  node_df <- layout$nodes
  # C++ uses 0-indexed node_idx, convert to 1-indexed for R
  node_df$node_idx <- node_df$node_idx + 1L
  edge_df <- layout$edges

  # Collect all histogram bars, cutpoint lines, and axis ticks
  all_bars <- list()
  all_thr_segs <- list()
  all_ticks <- list()
  node_bg_df <- list()
  leaf_bg_df <- list()
  leaf_label_df <- list()
  proj_label_df <- list()

  for (i in seq_len(nrow(node_df))) {
    row <- node_df[i, ]
    nd <- nodes[[row$node_idx]]

    if (nd$is_leaf) {
      predicted <- group_labels[nd$value]
      leaf_bg_df[[length(leaf_bg_df) + 1L]] <- data.frame(
        xmin = row$x - leaf_w / 2, xmax = row$x + leaf_w / 2,
        ymin = row$y - leaf_h / 2, ymax = row$y + leaf_h / 2,
        group = predicted,
        stringsAsFactors = FALSE
      )
      leaf_label_df[[length(leaf_label_df) + 1L]] <- data.frame(
        x = row$x, y = row$y, label = predicted,
        stringsAsFactors = FALSE
      )
    } else {
      node_bg_df[[length(node_bg_df) + 1L]] <- data.frame(
        xmin = row$x - node_w / 2, xmax = row$x + node_w / 2,
        ymin = row$y - node_h / 2, ymax = row$y + node_h / 2,
        stringsAsFactors = FALSE
      )

      hist_data <- compute_histogram_bars(nd, row$x, row$y, node_w, node_h, group_labels)

      if (!is.null(hist_data)) {
        all_bars[[length(all_bars) + 1L]] <- hist_data$bars
        all_thr_segs[[length(all_thr_segs) + 1L]] <- data.frame(
          x = hist_data$thr_x, xend = hist_data$thr_x,
          y = hist_data$thr_y_min, yend = hist_data$thr_y_max,
          stringsAsFactors = FALSE
        )
        all_ticks[[length(all_ticks) + 1L]] <- hist_data$ticks
      }

      proj_label_df[[length(proj_label_df) + 1L]] <- data.frame(
        x = row$x,
        y = row$y - node_h / 2 - 0.15,
        label = format_projector(nd$projector, var_names, max_terms),
        stringsAsFactors = FALSE
      )
    }
  }

  bar_df <- if (length(all_bars) > 0) do.call(rbind, all_bars) else NULL
  thr_df <- if (length(all_thr_segs) > 0) do.call(rbind, all_thr_segs) else NULL
  tick_df <- if (length(all_ticks) > 0) do.call(rbind, all_ticks) else NULL
  node_bg <- if (length(node_bg_df) > 0) do.call(rbind, node_bg_df) else NULL
  leaf_bg <- if (length(leaf_bg_df) > 0) do.call(rbind, leaf_bg_df) else NULL
  leaf_labels <- if (length(leaf_label_df) > 0) do.call(rbind, leaf_label_df) else NULL
  proj_labels <- if (length(proj_label_df) > 0) do.call(rbind, proj_label_df) else NULL

  p <- ggplot2::ggplot() +
    ggplot2::theme_void()

  # Edges
  if (nrow(edge_df) > 0) {
    p <- p + ggplot2::geom_segment(
      data = edge_df,
      ggplot2::aes(x = from_x, y = from_y, xend = to_x, yend = to_y),
      color = ppforest2_col_edge(), linewidth = ppforest2_lw_light()
    )
    # Edge labels at midpoint
    edge_label_df <- data.frame(
      x = (edge_df$from_x + edge_df$to_x) / 2,
      y = (edge_df$from_y + edge_df$to_y) / 2,
      label = edge_df$edge_label,
      stringsAsFactors = FALSE
    )
    p <- p + ggplot2::geom_label(
      data = edge_label_df,
      ggplot2::aes(x = x, y = y, label = label),
      size = ppforest2_text_edge(), label.size = 0, fill = "white", alpha = 0.8,
      label.padding = ggplot2::unit(0.15, "lines")
    )
  }

  # Node backgrounds
  if (!is.null(node_bg)) {
    p <- p + ggplot2::geom_rect(
      data = node_bg,
      ggplot2::aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
      fill = "white", color = ppforest2_col_border(), linewidth = ppforest2_lw_light()
    )
  }

  # Histogram bars
  if (!is.null(bar_df)) {
    p <- p + ggplot2::geom_rect(
      data = bar_df,
      ggplot2::aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax,
                   fill = group),
      alpha = ppforest2_alpha_hist(), color = NA
    )
  }

  # Threshold lines
  if (!is.null(thr_df)) {
    p <- p + ggplot2::geom_segment(
      data = thr_df,
      ggplot2::aes(x = x, y = y, xend = xend, yend = yend),
      linetype = "dashed", color = ppforest2_col_cutpoint(), linewidth = ppforest2_lw_medium()
    )
  }

  # Axis ticks
  if (!is.null(tick_df)) {
    p <- p + ggplot2::geom_text(
      data = tick_df,
      ggplot2::aes(x = x, y = y, label = label),
      size = ppforest2_text_tick(), vjust = 1.5, color = ppforest2_col_tick()
    )
  }

  # Leaf backgrounds
  if (!is.null(leaf_bg)) {
    p <- p + ggplot2::geom_rect(
      data = leaf_bg,
      ggplot2::aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax,
                   fill = group),
      alpha = ppforest2_alpha_leaf(), color = ppforest2_col_border(), linewidth = ppforest2_lw_light()
    )
  }

  # Leaf labels
  if (!is.null(leaf_labels)) {
    p <- p + ggplot2::geom_text(
      data = leaf_labels,
      ggplot2::aes(x = x, y = y, label = label),
      size = ppforest2_text_leaf(), fontface = "bold"
    )
  }

  # Projector labels below internal nodes
  if (!is.null(proj_labels)) {
    p <- p + ggplot2::geom_text(
      data = proj_labels,
      ggplot2::aes(x = x, y = y, label = label),
      size = ppforest2_text_proj(), vjust = 1, color = ppforest2_col_tick()
    )
  }

  # Color scale, title, and legend
  p <- p +
    ggplot2::scale_fill_manual(values = group_colors, name = "Class") +
    ggplot2::ggtitle("Tree Structure") +
    ggplot2::coord_cartesian(
      xlim = c(min(node_df$x) - node_w / 2 - 0.2,
               max(node_df$x) + node_w / 2 + 0.2),
      ylim = c(min(node_df$y) - node_h / 2 - 0.3,
               max(node_df$y) + node_h / 2 + 0.1)
    ) +
    ggplot2::theme(
      plot.title       = ggplot2::element_text(hjust = 0, margin = ggplot2::margin(b = 5)),
      plot.title.position = "plot",
      legend.position  = "top",
      legend.justification = "left",
      legend.margin    = ggplot2::margin(t = 0, b = 5),
      plot.margin      = ggplot2::margin(2, 5, 2, 5)
    )

  p
}
