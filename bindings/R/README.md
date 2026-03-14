# ppforest2

<!-- badges: start -->
[![R-CMD-check](https://github.com/andres-vidal/ppforest2/actions/workflows/run-r-check.yml/badge.svg)](https://github.com/andres-vidal/ppforest2/actions/workflows/run-r-check.yml)
<!-- badges: end -->

ppforest2 provides projection pursuit oblique decision trees and random
forests for classification.  Instead of splitting on single variables,
each node projects the data onto a linear combination of features,
capturing structure that axis-aligned trees miss.

The package wraps a high-performance C++ core and is intended as a modern
successor to [`PPforest`](https://cran.r-project.org/package=PPforest).

Key capabilities: oblique splits via projection pursuit, multi-threaded
forest training (OpenMP), cross-platform reproducibility, three variable
importance measures (projection-based, weighted, permutation), LDA/PDA
optimisation, OOB error estimation, and
[parsnip](https://parsnip.tidymodels.org/) / tidymodels integration.

## Installation

```r
# install.packages("devtools")
devtools::install_github("andres-vidal/ppforest2", subdir = "bindings/R", build = FALSE)
```

## Usage

### Single tree

```r
library(ppforest2)

model <- pptr(Species ~ ., data = iris)
predict(model, iris[1:5, ])
summary(model)
```

### Random forest

```r
forest <- pprf(Species ~ ., data = iris, size = 500)
predict(forest, iris[1:5, ])
predict(forest, iris[1:5, ], type = "prob")   # vote proportions
summary(forest)
```

### Regularisation (PDA)

When classes are highly correlated or the number of variables is large
relative to the sample size, penalised discriminant analysis can improve
separation:

```r
pptr(Species ~ ., data = iris, lambda = 0.5)
```

### Visualisation

ppforest2 provides four diagnostic plot types (requires
[ggplot2](https://ggplot2.tidyverse.org/)):

```r
# Mosaic overview: structure + importance + boundaries
plot(model)

# Individual plot types
plot(model, type = "structure")     # tree diagram with per-node histograms
plot(model, type = "importance")    # variable importance bar chart
plot(model, type = "projection")    # projected data at each split
plot(model, type = "boundaries")    # decision boundaries in feature space

# Forest: importance across all trees, or inspect individual trees
plot(forest)
plot(forest, type = "structure", tree_index = 1)
plot(forest, type = "boundaries", tree_index = 1)
```

### tidymodels integration

ppforest2 integrates with [parsnip](https://parsnip.tidymodels.org/):

```r
library(parsnip)

# Single tree
spec <- pp_tree(lambda = 0) |> set_engine("ppforest2") |> set_mode("classification")
fit  <- fit(spec, Species ~ ., data = iris)

# Random forest
spec <- pp_rand_forest(trees = 50, mtry = 2) |> set_engine("ppforest2")
fit  <- spec |> fit(Species ~ ., data = iris)
predict(fit, iris, type = "prob")
```

### JSON serialisation

Models can be saved and loaded in JSON format, enabling interoperability
with the C++ CLI and other language bindings:

```r
save_json(model, "model.json")
restored <- load_json("model.json")
```

## Learning more

- `vignette("introduction")` — a tutorial covering trees, forests,
  visualisation, and tidymodels integration.
- [C++ API Reference](https://andres-vidal.github.io/ppforest2/main/cpp/) —
  core library documentation (Doxygen).
- [GitHub repository](https://github.com/andres-vidal/ppforest2) —
  source code, build instructions, and benchmarks.
