Sys.setenv(R_TESTS = "")
Sys.setenv(OMP_THREAD_LIMIT = "1")
Sys.setenv(OMP_NUM_THREADS = "1")

library(testthat)
library(ppforest2)

if (!requireNamespace("jsonlite", quietly = TRUE)) {
  stop("jsonlite is required for reproducibility tests")
}

golden_path <- function(...) {
  system.file("golden", ..., package = "ppforest2")
}

load_golden <- function(...) {
  path <- golden_path(...)
  if (path == "" || !file.exists(path)) {
    stop(paste("Golden file not found:", ...))
  }
  # simplifyVector = FALSE preserves nested tree structure for forests.
  # Flat arrays are converted to vectors/matrices in comparison helpers.
  jsonlite::fromJSON(path, simplifyVector = FALSE)
}

# Convert a JSON list (from simplifyVector=FALSE) to a numeric/integer vector.
as_vec <- function(x) unlist(x)

# Convert a JSON list-of-lists to a matrix (row-major).
as_mat <- function(x) {
  do.call(rbind, lapply(x, unlist))
}

# Golden files use C++ group ordering (order of first appearance in CSV).
# R uses alphabetical factor levels by default. To ensure the R model is
# trained on identically-ordered data, we reorder the factor levels to
# match the golden file before training.
prepare_data <- function(data, golden, response = "Type") {
  # Response column defaults to "Type" (the convention for ppforest2's bundled
  # classification datasets) but is overridable so base R's `iris` (response
  # column "Species") works without a rename shim at every call site.
  golden_groups <- as_vec(golden$meta$groups)
  data[[response]] <- factor(data[[response]], levels = golden_groups)
  data
}

# ---------------------------------------------------------------------------
# Config (strategy) comparison
# ---------------------------------------------------------------------------

compare_strategy <- function(spec_strategy, config_strategy) {
  # Compare all fields from the golden config against the training spec.
  # The R training spec has an extra display_name field from wrap(); ignore it.
  for (key in names(config_strategy)) {
    expect_equal(spec_strategy[[key]], config_strategy[[key]], tolerance = 1e-5, info = paste("strategy field:", key))
  }
}

compare_config <- function(model, golden) {
  config <- golden$config
  spec <- model$training_spec

  # Strategies
  compare_strategy(spec$pp, config$pp)
  compare_strategy(spec$vars, config$vars)
  compare_strategy(spec$cutpoint, config$cutpoint)
  compare_strategy(spec$stop, config$stop)
  compare_strategy(spec$binarize, config$binarize)
  compare_strategy(spec$grouping, config$grouping)
  compare_strategy(spec$leaf, config$leaf)

  # Scalar fields
  expect_equal(spec$size, config$size)
  expect_equal(spec$seed, config$seed)
  expect_equal(spec$max_retries, config$max_retries)
}

# ---------------------------------------------------------------------------
# Model structure comparison
# ---------------------------------------------------------------------------

# Normalize a node to a canonical representation for comparison.
# R model nodes use 1-based group indices; golden JSON uses string labels.
# R indices are mapped to string labels via group_names for comparison.
normalize_node <- function(node, group_names = NULL) {
  if (!is.null(node$value)) {
    value <- if (!is.null(group_names)) {
      group_names[as.integer(node$value)]
    } else {
      node$value
    }

    list(value = value)
  } else {
    groups <- if (!is.null(group_names)) {
      sort(group_names[as.integer(unlist(node$groups))])
    } else {
      sort(as.character(unlist(node$groups)))
    }

    list(
      projector      = as.numeric(unlist(node$projector)),
      cutpoint      = as.numeric(node$cutpoint),
      pp_index_value = as.numeric(node$pp_index_value),
      groups         = groups,
      lower          = normalize_node(node$lower, group_names),
      upper          = normalize_node(node$upper, group_names)
    )
  }
}

compare_tree <- function(actual_root, expected_root, group_names, info = "") {
  actual <- normalize_node(actual_root, group_names = group_names)
  expected <- normalize_node(expected_root, group_names = NULL)
  expect_equal(actual, expected, tolerance = 1e-3, info = info)
}

# ---------------------------------------------------------------------------
# Output comparison helpers
# ---------------------------------------------------------------------------

compare_predictions <- function(model, golden, data) {
  predicted <- predict(model, data)
  golden_preds <- as.character(as_vec(golden$predictions))
  expected <- factor(golden_preds, levels = levels(predicted))
  expect_equal(predicted, expected)
}

compare_error_rate <- function(model, golden, data, response) {
  predicted <- predict(model, data)
  error_rate <- mean(predicted != response)
  expect_equal(error_rate, golden$error_rate, tolerance = 1e-3)
}

compare_confusion_matrix <- function(model, golden, data, response) {
  predicted <- predict(model, data)
  expected_cm <- golden$training_metrics$confusion_matrix
  golden_groups <- as.character(as_vec(golden$meta$groups))
  cm_labels <- as.character(as_vec(expected_cm$labels))

  # The saved CM is dimensioned by `union(predictions, actual)` on the
  # C++ side. In normal operation predictions ⊆ groups and actual ⊆
  # groups, so the CM labels equal the meta.groups set — assert that
  # invariant explicitly so any future "prediction-only label" case
  # surfaces here instead of silently producing a misaligned matrix.
  expect_equal(cm_labels, golden_groups, info = "CM label set must equal meta.groups (no prediction-only labels)")

  n_groups <- length(model$groups)
  # Align predicted levels with response levels (golden group ordering)
  predicted <- factor(predicted, levels = levels(response))
  actual_idx <- as.integer(response)
  pred_idx <- as.integer(predicted)

  cm <- matrix(0L, nrow = n_groups, ncol = n_groups)
  for (i in seq_along(actual_idx)) {
    cm[actual_idx[i], pred_idx[i]] <- cm[actual_idx[i], pred_idx[i]] + 1L
  }

  expect_equal(cm, as_mat(expected_cm$matrix))

  group_errors <- numeric(n_groups)
  for (i in seq_len(n_groups)) {
    row_total <- sum(cm[i, ])
    if (row_total > 0) {
      group_errors[i] <- 1 - cm[i, i] / row_total
    }
  }
  expect_equal(group_errors, as.numeric(as_vec(expected_cm$group_errors)),
    tolerance = 1e-3)
}

compare_vote_proportions <- function(model, golden, data) {
  probs <- predict(model, data, type = "prob")
  actual <- unname(as.matrix(probs))
  expect_equal(actual, as_mat(golden$vote_proportions), tolerance = 1e-3)
}

compare_vi <- function(model, golden, key, r_field) {
  expected <- as.numeric(as_vec(golden$variable_importance[[key]]))
  # The eager fields on model$vi are `scale` and `projections`; weighted and
  # permuted importances are lazy accessors.
  actual <- switch(r_field,
    weighted = weighted_importance(model),
    permuted = permuted_importance(model),
    model$vi[[r_field]]
  )
  expect_equal(actual, expected, tolerance = 1e-3)
}

# ---------------------------------------------------------------------------
# Tree: iris tree-pda-s0
# ---------------------------------------------------------------------------

describe("Reproducibility: iris tree-pda-s0", {
  golden <- load_golden("iris", "tree-pda-s0.json")
  d <- prepare_data(iris, golden, response = "Species")
  model <- pptr(Species ~ ., data = d, lambda = 0, seed = 0)

  it("training config matches golden file", {
    compare_config(model, golden)
  })

  it("model structure matches golden file", {
    compare_tree(model$root, golden$model$root, as_vec(golden$meta$groups))
  })

  it("predictions match golden file", {
    compare_predictions(model, golden, d)
  })

  it("error rate matches golden file", {
    compare_error_rate(model, golden, d, d$Species)
  })

  it("confusion matrix matches golden file", {
    compare_confusion_matrix(model, golden, d, d$Species)
  })

  it("VI projections match golden file", {
    compare_vi(model, golden, "projections", "projections")
  })
})

# ---------------------------------------------------------------------------
# Tree: crab tree-pda-s0
# ---------------------------------------------------------------------------

describe("Reproducibility: crab tree-pda-s0", {
  golden <- load_golden("crab", "tree-pda-s0.json")
  d <- prepare_data(crab, golden)
  model <- pptr(Type ~ ., data = d, lambda = 0, seed = 0)

  it("training config matches golden file", {
    compare_config(model, golden)
  })

  it("model structure matches golden file", {
    compare_tree(model$root, golden$model$root, as_vec(golden$meta$groups))
  })

  it("predictions match golden file", {
    compare_predictions(model, golden, d)
  })

  it("error rate matches golden file", {
    compare_error_rate(model, golden, d, d$Type)
  })

  it("confusion matrix matches golden file", {
    compare_confusion_matrix(model, golden, d, d$Type)
  })

  it("VI projections match golden file", {
    compare_vi(model, golden, "projections", "projections")
  })
})

# ---------------------------------------------------------------------------
# Forest: iris forest-pda-n5-s0
# ---------------------------------------------------------------------------

describe("Reproducibility: iris forest-pda-n5-s0", {
  golden <- load_golden("iris", "forest-pda-n5-s0.json")
  d <- prepare_data(iris, golden, response = "Species")
  model <- pprf(Species ~ ., data = d, size = 5, n_vars = 2, lambda = 0, seed = 0, threads = 1)

  it("training config matches golden file", {
    compare_config(model, golden)
  })

  it("model structure matches golden file", {
    for (i in seq_along(model$trees)) {
      compare_tree(model$trees[[i]]$root, golden$model$trees[[i]]$root,
        as_vec(golden$meta$groups), info = paste("tree", i))
    }
  })

  it("predictions match golden file", {
    compare_predictions(model, golden, d)
  })

  it("error rate matches golden file", {
    compare_error_rate(model, golden, d, d$Species)
  })

  it("confusion matrix matches golden file", {
    compare_confusion_matrix(model, golden, d, d$Species)
  })

  it("OOB error matches golden file", {
    expect_equal(oob_error(model), golden$oob_metrics$error_rate, tolerance = 1e-3)
  })

  it("VI projections match golden file", {
    compare_vi(model, golden, "projections", "projections")
  })

  it("VI weighted projections match golden file", {
    compare_vi(model, golden, "weighted_projections", "weighted")
  })

  it("VI permuted match golden file", {
    compare_vi(model, golden, "permuted", "permuted")
  })

  it("vote proportions match golden file", {
    compare_vote_proportions(model, golden, d)
  })
})

# ---------------------------------------------------------------------------
# Forest: iris forest-pda-l05-n5-s0
# ---------------------------------------------------------------------------

describe("Reproducibility: iris forest-pda-l05-n5-s0", {
  golden <- load_golden("iris", "forest-pda-l05-n5-s0.json")
  d <- prepare_data(iris, golden, response = "Species")
  model <- pprf(Species ~ ., data = d, size = 5, n_vars = 2, lambda = 0.5, seed = 0, threads = 1)

  it("training config matches golden file", {
    compare_config(model, golden)
  })

  it("model structure matches golden file", {
    for (i in seq_along(model$trees)) {
      compare_tree(model$trees[[i]]$root, golden$model$trees[[i]]$root,
        as_vec(golden$meta$groups), info = paste("tree", i))
    }
  })

  it("predictions match golden file", {
    compare_predictions(model, golden, d)
  })

  it("error rate matches golden file", {
    compare_error_rate(model, golden, d, d$Species)
  })

  it("confusion matrix matches golden file", {
    compare_confusion_matrix(model, golden, d, d$Species)
  })

  it("OOB error matches golden file", {
    expect_equal(oob_error(model), golden$oob_metrics$error_rate, tolerance = 1e-3)
  })

  it("VI projections match golden file", {
    compare_vi(model, golden, "projections", "projections")
  })

  it("VI weighted projections match golden file", {
    compare_vi(model, golden, "weighted_projections", "weighted")
  })

  it("VI permuted match golden file", {
    compare_vi(model, golden, "permuted", "permuted")
  })

  it("vote proportions match golden file", {
    compare_vote_proportions(model, golden, d)
  })
})

# ---------------------------------------------------------------------------
# Forest: crab forest-pda-n10-s0
# ---------------------------------------------------------------------------

describe("Reproducibility: crab forest-pda-n10-s0", {
  golden <- load_golden("crab", "forest-pda-n10-s0.json")
  d <- prepare_data(crab, golden)
  model <- pprf(Type ~ ., data = d, size = 10, n_vars = 3, lambda = 0, seed = 0, threads = 1)

  it("training config matches golden file", {
    compare_config(model, golden)
  })

  it("model structure matches golden file", {
    for (i in seq_along(model$trees)) {
      compare_tree(model$trees[[i]]$root, golden$model$trees[[i]]$root,
        as_vec(golden$meta$groups), info = paste("tree", i))
    }
  })

  it("predictions match golden file", {
    compare_predictions(model, golden, d)
  })

  it("error rate matches golden file", {
    compare_error_rate(model, golden, d, d$Type)
  })

  it("confusion matrix matches golden file", {
    compare_confusion_matrix(model, golden, d, d$Type)
  })

  it("OOB error matches golden file", {
    expect_equal(oob_error(model), golden$oob_metrics$error_rate, tolerance = 1e-3)
  })

  it("VI projections match golden file", {
    compare_vi(model, golden, "projections", "projections")
  })

  it("VI weighted projections match golden file", {
    compare_vi(model, golden, "weighted_projections", "weighted")
  })

  it("VI permuted match golden file", {
    compare_vi(model, golden, "permuted", "permuted")
  })

  it("vote proportions match golden file", {
    compare_vote_proportions(model, golden, d)
  })
})

# ---------------------------------------------------------------------------
# Forest: wine forest-pda-n10-s0
# ---------------------------------------------------------------------------

describe("Reproducibility: wine forest-pda-n10-s0", {
  golden <- load_golden("wine", "forest-pda-n10-s0.json")
  d <- prepare_data(wine, golden)
  model <- pprf(Type ~ ., data = d, size = 10, n_vars = 4, lambda = 0, seed = 0, threads = 1)

  it("training config matches golden file", {
    compare_config(model, golden)
  })

  it("model structure matches golden file", {
    for (i in seq_along(model$trees)) {
      compare_tree(model$trees[[i]]$root, golden$model$trees[[i]]$root,
        as_vec(golden$meta$groups), info = paste("tree", i))
    }
  })

  it("predictions match golden file", {
    compare_predictions(model, golden, d)
  })

  it("error rate matches golden file", {
    compare_error_rate(model, golden, d, d$Type)
  })

  it("confusion matrix matches golden file", {
    compare_confusion_matrix(model, golden, d, d$Type)
  })

  it("OOB error matches golden file", {
    expect_equal(oob_error(model), golden$oob_metrics$error_rate, tolerance = 1e-3)
  })

  it("VI projections match golden file", {
    compare_vi(model, golden, "projections", "projections")
  })

  it("VI weighted projections match golden file", {
    compare_vi(model, golden, "weighted_projections", "weighted")
  })

  it("VI permuted match golden file", {
    compare_vi(model, golden, "permuted", "permuted")
  })

  it("vote proportions match golden file", {
    compare_vote_proportions(model, golden, d)
  })
})

# ---------------------------------------------------------------------------
# Forest: glass forest-pda-n10-s0
# ---------------------------------------------------------------------------

describe("Reproducibility: glass forest-pda-n10-s0", {
  golden <- load_golden("glass", "forest-pda-n10-s0.json")
  d <- prepare_data(glass, golden)
  model <- pprf(Type ~ ., data = d, size = 10, n_vars = 3, lambda = 0, seed = 0, threads = 1)

  it("training config matches golden file", {
    compare_config(model, golden)
  })

  it("model structure matches golden file", {
    for (i in seq_along(model$trees)) {
      compare_tree(model$trees[[i]]$root, golden$model$trees[[i]]$root,
        as_vec(golden$meta$groups), info = paste("tree", i))
    }
  })

  it("predictions match golden file", {
    compare_predictions(model, golden, d)
  })

  it("error rate matches golden file", {
    compare_error_rate(model, golden, d, d$Type)
  })

  it("confusion matrix matches golden file", {
    compare_confusion_matrix(model, golden, d, d$Type)
  })

  it("OOB error matches golden file", {
    expect_equal(oob_error(model), golden$oob_metrics$error_rate, tolerance = 1e-3)
  })

  it("VI projections match golden file", {
    compare_vi(model, golden, "projections", "projections")
  })

  it("VI weighted projections match golden file", {
    compare_vi(model, golden, "weighted_projections", "weighted")
  })

  it("VI permuted match golden file", {
    compare_vi(model, golden, "permuted", "permuted")
  })

  it("vote proportions match golden file", {
    compare_vote_proportions(model, golden, d)
  })
})
