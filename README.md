# ppforest2

[![C++ Tests](https://github.com/andres-vidal/ppforest2/actions/workflows/run-test.yml/badge.svg)](https://github.com/andres-vidal/ppforest2/actions/workflows/run-test.yml)
[![R Package Check](https://github.com/andres-vidal/ppforest2/actions/workflows/run-r-check.yml/badge.svg)](https://github.com/andres-vidal/ppforest2/actions/workflows/run-r-check.yml)

> **Work in progress** — this repository contains ongoing research and development work. Interfaces and behavior are expected to evolve as the project matures.

**ppforest2** is a fast, memory-efficient implementation of
[Projection Pursuit Random Forests](https://www.tandfonline.com/doi/full/10.1080/10618600.2020.1870480),
built on
[Projection Pursuit (oblique) Decision Trees](https://projecteuclid.org/journals/electronic-journal-of-statistics/volume-7/issue-none/PPtree-Projection-pursuit-classification-tree/10.1214/13-EJS810.full).
By learning linear projections at each split, the model captures complex structure that axis-aligned trees often miss, without sacrificing interpretability or scalability.

The project provides a high-performance C++ core with interfaces for **R** and a command-line interface (CLI), with **Python** bindings planned. In the R ecosystem, it is intended as a modern successor to
[`PPforest`](https://cran.r-project.org/web/packages/PPforest/index.html),
offering the same statistical foundations with significantly improved computational performance.

Developed as a Bachelor's thesis project in Statistics at **Universidad de la República (Uruguay)**.

**Key capabilities:** oblique splits via projection pursuit, multi-threaded forest training (OpenMP), cross-platform reproducibility with golden tests, multiple variable importance measures (projection-based, weighted, permutation), LDA/PDA optimization, OOB error estimation, and [parsnip](https://parsnip.tidymodels.org/) / tidymodels integration.

**Documentation:** [andres-vidal.github.io/ppforest2](https://andres-vidal.github.io/ppforest2/) — [C++ API Reference](https://andres-vidal.github.io/ppforest2/main/cpp/) (Doxygen) · [R Package Reference](https://andres-vidal.github.io/ppforest2/main/r/) (pkgdown)

## Contents

- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Building and Testing](#building-and-testing)
- [Benchmarking](#benchmarking)
- [R Package](#r-package)
- [Documentation](#documentation)
- [Reproducibility Break Protocol](#reproducibility-break-protocol)
- [Versioning](#versioning)
- [License](#license)

## Quick Start

### CLI

Build and use the `ppforest2` command-line tool directly:

```bash
# Compile the project into the .build folder, where the `ppforest2` executable is.
make build

# Train a forest on a CSV dataset and save the model
ppforest2 train --data data.csv --trees 100 --lambda 0.5 --save model.json

# Predict on new data using a saved model
ppforest2 predict --model model.json --data test.csv

# Evaluate with smart convergence (default)
ppforest2 evaluate --data data.csv --trees 100 --train-ratio 0.7

# Evaluate with fixed iterations (disables convergence)
ppforest2 evaluate --data data.csv --trees 100 -i 10 --train-ratio 0.7

# Evaluate on simulated data (1000 rows, 10 features, 3 groups)
ppforest2 evaluate --simulate 1000x10x3 --trees 50

# Run performance benchmarks across scenarios
ppforest2 benchmark -s bench/default-scenarios.json
```

### R

Install the R package (CRAN submission is planned once the package stabilizes):

```r
# install.packages("devtools")
devtools::install_github("andres-vidal/ppforest2", subdir = "bindings/R", build = FALSE)
```

```r
library(ppforest2)

# Single projection pursuit tree
model <- pptr(Type ~ ., data = iris)
predict(model, iris)

# Random forest with 50 trees
forest <- pprf(Type ~ ., data = iris, size = 50)
predict(forest, iris)
predict(forest, iris, type = "prob")   # vote proportions
summary(forest)                        # variable importance & model info
```

Visualize models (requires `ggplot2`):

```r
# Mosaic overview: tree structure + variable importance + decision boundaries
plot(model)

# Individual plot types
plot(model, type = "structure")     # tree diagram with histograms at each node
plot(model, type = "importance")    # variable importance bar chart
plot(model, type = "projection")    # projected data at the root split
plot(model, type = "boundaries")    # decision boundaries in feature space

# Decision boundaries adapt to the number of features:
#   - 1 feature:  number-line plot with colored decision regions
#   - 2 features: scatterplot with polygon regions and boundary lines
#   - 3+ features: pairwise scatterplot matrix (lower triangle)
model2 <- pptr(x = iris[, 1:2], y = iris$Type)
plot(model2, type = "boundaries")   # 2D boundary plot

# Forest: importance across all trees, or inspect individual trees
plot(forest)                                         # variable importance
plot(forest, type = "structure", tree_index = 1)     # structure of tree #1
plot(forest, type = "boundaries", tree_index = 1)    # boundaries of tree #1
```

Works with [parsnip](https://parsnip.tidymodels.org/) / tidymodels:

```r
library(parsnip)

spec <- pp_rand_forest(trees = 50, mtry = 2) %>% set_engine("ppforest2")
fit <- spec %>% fit(Type ~ ., data = iris)
predict(fit, iris, type = "prob")
```

## CLI Reference

The `ppforest2` command-line tool provides four subcommands for training, prediction, evaluation, and benchmarking. After `make build`, the binary is available at `.build/ppforest2`.

### Global Options

| Flag              | Description                                      |
|-------------------|--------------------------------------------------|
| `--version, -V`   | Print version and exit                           |
| `--quiet, -q`     | Suppress all terminal output                     |
| `--no-color`      | Disable colored output                           |
| `--config <file>` | Read parameters from a JSON config file          |

Config files accept both `snake_case` and `kebab-case` keys. Top-level keys are broadcast to all subcommands:

```json
{ "trees": 100, "lambda": 0.5, "train": { "data": "iris.csv" } }
```

### `train` — Train a Model

Train a single tree or forest on a CSV dataset and save the result.

```bash
ppforest2 train -d data.csv -t 100 -l 0.5 -s model.json
ppforest2 train -d data.csv -t 0                # single tree (no forest)
ppforest2 train -d data.csv --no-save            # train without saving
ppforest2 train -d data.csv --no-metrics         # skip variable importance
```

| Flag                     | Default       | Description                                       |
|--------------------------|---------------|---------------------------------------------------|
| `-d, --data <file>`      | *(required)*  | CSV training data                                 |
| `-t, --trees <N>`        | `100`         | Number of trees (`0` for a single tree)           |
| `-l, --lambda <X>`       | `0.5`         | PDA penalty; `0` = LDA, `(0,1]` = PDA             |
| `-r, --seed <N>`         | *(random)*    | Random seed for reproducibility                   |
| `-v, --vars <spec>`      | `0.5`         | Features per split (see [Variable selection](#variable-selection)) |
| `--threads <N>`          | *(all cores)* | Number of OpenMP threads                          |
| `--max-retries <N>`      | `3`           | Max retries for degenerate trees                  |
| `-s, --save <file>`      | `model.json`  | Output model path (`.json` added if missing)      |
| `--no-save`              | —             | Skip saving the model                             |
| `--no-metrics`           | —             | Skip variable importance computation              |

The saved model JSON includes the full serialization, training configuration, variable importance metrics, and OOB error (forests only).

### `predict` — Predict with a Saved Model

Load a trained model and classify new observations.

```bash
ppforest2 predict -M model.json -d test.csv
ppforest2 predict -M model.json -d test.csv -o predictions.json
ppforest2 predict -M model.json -d test.csv --no-proportions -o predictions.json
```

| Flag                     | Default       | Description                                       |
|--------------------------|---------------|---------------------------------------------------|
| `-M, --model <file>`     | *(required)*  | Saved model JSON                                  |
| `-d, --data <file>`      | *(required)*  | CSV data to classify                              |
| `-o, --output <file>`    | —             | Save predictions, error rate, and confusion matrix to JSON |
| `--no-metrics`           | —             | Omit error rate and confusion matrix from output  |
| `--no-proportions`       | —             | Omit vote proportions from output (forest only)   |

If the CSV includes response labels, the tool reports the error rate and confusion matrix. For forest models, the JSON output includes per-group vote proportions by default; use `--no-proportions` to omit them.

### `evaluate` — Train-Test Evaluation

Split data into training and test sets, train a model, and measure performance. Supports smart convergence for stable timing measurements or a fixed number of iterations.

```bash
# Evaluate on a CSV file
ppforest2 evaluate -d data.csv -t 50 -p 0.7

# Evaluate on simulated data (1000 rows, 10 features, 3 groups)
ppforest2 evaluate --simulate 1000x10x3 -t 50

# Fixed iterations (disables convergence)
ppforest2 evaluate -d data.csv -t 50 -i 20
```

**Data source** (mutually exclusive):

| Flag                     | Description                                       |
|--------------------------|---------------------------------------------------|
| `-d, --data <file>`      | CSV file                                          |
| `--simulate NxMxK`       | Generate synthetic data (rows × features × groups) |

**Simulation parameters** (only with `--simulate`):

| Flag                            | Default  | Description                           |
|---------------------------------|----------|---------------------------------------|
| `--simulate-mean <X>`           | `100.0`  | Feature mean                          |
| `--simulate-mean-separation <X>`| `50.0`   | Mean separation between groups        |
| `--simulate-sd <X>`             | `10.0`   | Standard deviation                    |

**Iteration control:**

| Flag                     | Default  | Description                                       |
|--------------------------|----------|---------------------------------------------------|
| `-p, --train-ratio <X>`  | `0.7`    | Proportion of data used for training              |
| `-i, --iterations <N>`   | —        | Fixed iteration count (disables convergence)      |
| `--warmup <N>`           | `0`      | Warmup iterations discarded before measuring      |
| `-o, --output <file>`    | —        | Save results to JSON                              |
| `-e, --export <dir>`     | —        | Export experiment bundle (config + data + results) |

**Convergence parameters** (active when `-i` is not set):

| Flag                        | Default  | Description                                       |
|-----------------------------|----------|---------------------------------------------------|
| `--convergence-max <N>`     | `200`    | Hard upper bound on iterations                    |
| `--convergence-cv <X>`      | `0.05`   | CV threshold (stop when std/mean < threshold)     |
| `--convergence-min <N>`     | `10`     | Minimum iterations before checking convergence    |
| `--convergence-window <N>`  | `3`      | Consecutive stable checks required to stop        |

All model parameters (`--trees`, `--lambda`, `--seed`, `--vars`, `--threads`, `--max-retries`) are also available.

### `benchmark` — Multi-Scenario Benchmarks

Run a suite of evaluation scenarios and report results in a table. Each scenario runs as a separate subprocess for accurate per-scenario memory measurement.

```bash
ppforest2 benchmark -s bench/default-scenarios.json
ppforest2 benchmark -s scenarios.json -b baseline.json       # compare against baseline
ppforest2 benchmark -s scenarios.json -o results.json --csv results.csv
ppforest2 benchmark -s scenarios.json --format markdown
```

| Flag                     | Default  | Description                                       |
|--------------------------|----------|---------------------------------------------------|
| `-s, --scenarios <file>` | *(required)* | JSON scenarios file                           |
| `-b, --baseline <file>`  | —        | Baseline results JSON for comparison              |
| `-o, --output <file>`    | —        | Save results to JSON                              |
| `--csv <file>`           | —        | Save results to CSV                               |
| `--format <fmt>`         | `table`  | Output format: `table` or `markdown`              |
| `-i, --iterations <N>`   | —        | Override iteration count for all scenarios        |
| `-p, --train-ratio <X>`  | —        | Override train ratio for all scenarios            |

When comparing against a baseline, delta columns show regressions and improvements with color indicators.

### Variable Selection

The `--vars` flag controls how many features are considered at each split in a forest. It accepts three formats:

| Format   | Example  | Meaning                         |
|----------|----------|---------------------------------|
| Integer  | `5`      | Use exactly 5 features          |
| Decimal  | `0.5`    | Use 50% of features             |
| Fraction | `1/3`    | Use one-third of features       |

This parameter is ignored for single trees (`--trees 0`), which always use all features.

## Architecture

The project is organized into a shared C++ core and language-specific bindings:

- **C++ core** (`core/`) — All models, training algorithms, statistics, serialization, and CLI live here. This is the single source of truth for the implementation. External dependencies (Eigen, nlohmann/json, pcg, GoogleTest, Google Benchmark, CLI11, fmt, csv-parser) are declared in `core/Dependencies.cmake` and fetched automatically via CMake `FetchContent`.

- **R package** (`bindings/R/`) — Thin Rcpp layer that exposes the C++ core to R. Type conversions between R and C++ types are defined in `inst/include/ppforest2.h`. Roxygen documentation and parsnip integration are R-only.

- **Visualization** (`core/src/models/Visualization.hpp/cpp` + `bindings/R/R/plot-*.R`) — Split between C++ and R. C++ handles geometry: tree traversal visitors collect per-node data, clip decision boundary lines via parametric line clipping, and compute convex decision region polygons via Sutherland–Hodgman polygon clipping. R handles rendering via ggplot2, translating the C++ output into layers and assembling composite layouts (mosaic, pairwise facets, tree diagrams). The tree structure visualization — with embedded per-node histograms and projector labels — is inspired by [dtreeviz](https://github.com/parrt/dtreeviz).

- **Numeric precision** — The C++ core uses single-precision (`float`) arithmetic by default. This is sufficient for most classification tasks and reduces memory usage. If you need higher precision (e.g., datasets with very large feature values or near-singular within-group covariance matrices), compile with `-DPPFOREST2_DOUBLE_PRECISION=ON` to switch to `double` throughout. The R package always uses whatever precision the core was compiled with.

- **Python bindings** — Planned.

### Design patterns

The C++ core uses two design patterns to keep the algorithm extensible without heavily modifying existing code:

- **Strategy** — The projection-pursuit optimization step (`PPStrategy`), dimensionality reduction step (`DRStrategy`), and split-point rule (`SRStrategy`) are each defined as abstract interfaces. Concrete implementations (e.g. `PPPDAStrategy`, `DRUniformStrategy`) are composed at runtime via `TrainingSpec`, so new optimization criteria or variable selection methods can be added without changing the tree-building logic.

- **Visitor** — `TreeNode::Visitor` dispatches over the two node types (internal `TreeCondition` and leaf `TreeResponse`) and `Model::Visitor` dispatches over `Tree` and `Forest`. This avoids `dynamic_cast` and keeps traversal logic (serialization, visualization layout, variable importance) decoupled from the model types themselves.

## Prerequisites

|              | Linux            | macOS           | Windows                              |
|--------------|------------------|-----------------|--------------------------------------|
| **C++ core** | `cmake` >= 3.20, `make`, `gcc` | `cmake` >= 3.20, `make`, `clang` | `cmake` >= 3.20, `make`, MinGW `gcc` |
| **R package**| `R` >= 3.5       | `R` >= 3.5      | `R` >= 3.5, `Rtools`                |
| **OpenMP** (optional) | Usually included with `gcc` | `brew install libomp` | Usually included with MinGW |
| **R docs**   | TeX distribution with `pdflatex` | TeX distribution with `pdflatex` | TeX distribution with `pdflatex` |

For the R package, the C++ compiler must match the one R was built with (`gcc` on Linux/Windows, `clang` on macOS). OpenMP is optional but recommended for multi-threaded forest training; without it, forests train on a single thread.

## Building and Testing

### C++ core

```bash
make build              # Release build (C++ core + CLI + tests)
make test               # Build and run C++ tests (GoogleTest)
make build-debug        # Debug build with AddressSanitizer
make test-debug         # Run debug tests
make clean              # Remove all build artifacts (.build/, .debug/, .r-build/)
```

The release build produces the `ppforest2` CLI binary and the `ppforest2-test` test runner in `.build/`. The debug build enables AddressSanitizer (on Linux) and runtime assertions.

### R package

```bash
make r-install-deps     # Install R package dependencies via pak
make r-build            # Prepare source and run R CMD build (produces tarball)
make r-check            # Build and run R CMD check on the tarball
make r-check-cran       # Same as r-check with --as-cran for CRAN submission
make r-install          # Build and install the package locally
make r-document         # Regenerate Roxygen man pages
make r-clean            # Remove R compilation byproducts
```

### Development tools

```bash
make install-tools      # Download uncrustify, cppcheck, doxygen into .tools/
make format             # Format C++ code (uncrustify)
make format-dry         # Check formatting without applying changes
make analyze            # Run static analysis (cppcheck)
```

### Golden tests

```bash
make golden-regen       # Regenerate golden reference files from current code
```

Golden files in `golden/` are pre-computed reference outputs verified on every platform in CI. If a code change intentionally alters model output, regenerate them and commit the updated files. See [Reproducibility Break Protocol](#reproducibility-break-protocol) for the full procedure.

### Documentation

```bash
make docs               # Build all documentation (landing page + C++ API + R pkgdown)
make docs-cpp           # Build C++ API docs only (Doxygen)
make docs-r             # Build R package site only (pkgdown)
```

## Benchmarking

Performance benchmarks run configurable scenarios on simulated or real data, measuring execution time and peak RSS. Each scenario runs as a separate process for accurate per-scenario memory measurement.

### Running Benchmarks

```bash
make benchmark                     # Run default scenarios, print table
make benchmark-save                # Run and save results to bench/results.json + bench/results.csv
make benchmark-compare             # Run and compare against saved results
make benchmark-vs REF=main         # Compare current branch against another ref (branch/tag/commit)
```

### CLI Usage

```bash
# Run scenarios from a JSON file
ppforest2 benchmark -s bench/default-scenarios.json

# Save results as JSON and CSV
ppforest2 benchmark -s bench/default-scenarios.json -o results.json --csv results.csv

# Compare against a baseline
ppforest2 benchmark -s bench/default-scenarios.json -b baseline.json

# Override iteration count (forces fixed mode)
ppforest2 benchmark -s bench/default-scenarios.json -i 5
```

### Smart Convergence

The `evaluate` subcommand uses smart convergence by default: it monitors the
coefficient of variation (CV = std/mean) of timing measurements and stops once
results are statistically stable.

The algorithm:
1. Run at least `--convergence-min` (default: 10) iterations before checking.
2. After each iteration, if CV < `--convergence-cv` threshold (default: 0.05), increment a stability counter; otherwise reset it.
3. Stop when the counter reaches `--convergence-window` (default: 3) consecutive checks.
4. Never exceed `--convergence-max` (default: 200).

Use `-i N` to disable convergence and run exactly N iterations instead.

```bash
# Default: smart convergence (runs until CV < 5%)
ppforest2 evaluate --simulate 1000x20x3 -t 50

# Stricter threshold with warmup
ppforest2 evaluate --simulate 1000x20x3 -t 50 --warmup 2 --convergence-cv 0.03

# Tune convergence parameters
ppforest2 evaluate --simulate 1000x20x3 -t 50 --convergence-min 20 --convergence-window 5

# Fixed iterations (disables convergence)
ppforest2 evaluate --simulate 1000x20x3 -t 50 -i 10
```

### Scenario Format

Scenarios are defined in JSON with shared defaults and per-scenario overrides:

```json
{
  "defaults": {
    "trees": 100, "lambda": 0.5, "vars": 0.5,
    "seed": 42, "warmup": 2,
    "convergence": { "cv": 0.05, "max": 200 }
  },
  "scenarios": [
    { "name": "small-forest",  "n": 200,  "p": 5,  "g": 2, "trees": 50 },
    { "name": "medium-forest", "n": 1000, "p": 20, "g": 3 },
    { "name": "fixed-5",       "n": 1000, "p": 20, "g": 3, "iterations": 5 },
    { "name": "data-iris",     "data": "data/iris.csv", "trees": 50 }
  ]
}
```

Scenarios support two data sources:
- **Simulated**: specify `n`, `p`, `g` to generate synthetic data via `--simulate NxPxG`.
- **Real data**: specify `"data": "path/to/file.csv"` to use a CSV file. Data dimensions (n, p, g) are derived automatically from the file. The `n`, `p`, `g` fields are ignored when `data` is set.

Setting `"iterations"` forces fixed mode for that scenario; otherwise, smart convergence is used.

## R Package

### Building from Source

Install R dependencies, then build:

```bash
make r-install-deps
```

```bash
make r-build            # Prepare source and run R CMD build
make r-check            # Run R CMD check on the built tarball
make r-install          # Run R CMD INSTALL on the built tarball
make r-document         # Regenerate Roxygen man pages
make r-clean            # Remove compilation byproducts
```

> **Important:** Always use `make r-build` before checking or installing. This target copies the C++ core source into the R package's `src/core/` so it can be compiled on install.

### Build Process

The R package wraps the C++ core via Rcpp. Because the core lives outside the R package directory, the build process assembles a self-contained source tarball that can be compiled anywhere.

All workflows share a single dependency cache in `.build/_deps/`, populated by `make fetch-deps`. Dependencies are fetched once and reused across both the C++ and R build pipelines.

#### Tarball pipeline (`make r-check`)

1. **`fetch-deps`** — Runs a core-only cmake configure in `.build/` to download dependencies (Eigen, nlohmann/json, pcg, csv-parser, fmt) via FetchContent. No compilation.

2. **`r-prepare`** — Copies the core source into `src/core/`, dependency headers (nlohmann/json, pcg) from `.build/_deps/` into `inst/include/`, and golden files into `inst/golden/`.

3. **`r-build`** — Regenerates `RcppExports.cpp`/`RcppExports.R` via `Rcpp::compileAttributes()`, and runs `R CMD build` to produce a source tarball.

4. **`configure` / `configure.win`** — During `R CMD check` or `R CMD INSTALL`, the configure script detects the build context and compiles the C++ core:
   - **Monorepo** (`../../core/` exists): delegates to `make r-build-core`, which uses cmake incremental builds in `.r-build/`. Used by `devtools::load_all()` and `install_github`.
   - **Tarball** (`src/core/` bundled): runs cmake directly on the bundled source. Used by `R CMD INSTALL` from a tarball.

   The `PPFOREST2_FETCH_CACHE` environment variable can point to pre-downloaded sources to avoid re-fetching.

#### Development workflow (`devtools::load_all()`)

For iterative development, the configure script detects the monorepo layout and delegates to `make r-build-core`, which compiles the C++ core into `.r-build/` using R's compiler. cmake incremental builds ensure only changed files are recompiled. The static library, headers, and core source are copied into the R package for linking.

```r
devtools::load_all("bindings/R")   # edit C++ -> reload -> test
devtools::test("bindings/R")        # run testthat suite
```

#### Compiler handling

On macOS, `R CMD config CXX17` may return the compiler with architecture flags (e.g., `clang++ -arch arm64`). Since CMake's `CMAKE_CXX_COMPILER` expects only the compiler path, the build splits this value: the first word becomes the compiler, and any remaining flags are appended to `CMAKE_CXX_FLAGS`. This splitting is applied in the root Makefile (`r-build-core`), `configure`, and `configure.win`.

### How `install_github` Works

`install_github` requires `build = FALSE` so that `R CMD INSTALL` runs directly on the source directory within the cloned monorepo (without `build = FALSE`, `R CMD build` creates an intermediate tarball that loses the monorepo context):

```r
devtools::install_github("andres-vidal/ppforest2", subdir = "bindings/R", build = FALSE)
```

The configure script detects `../../core/` and delegates to `make r-build-core`, which builds the C++ core via cmake and copies it into the package.

## Documentation

The project has a unified documentation site combining a static landing page, a C++ API reference (Doxygen), and R package documentation (pkgdown). The site is deployed to GitHub Pages with versioned directories for each branch and tag.

### Deployment

Documentation is automatically deployed to GitHub Pages on pushes to `main`, `next`, and version tags (`v*`). Each version gets its own directory:

```
/              Redirects to /main/
/main/         Latest from the main branch
/next/         Latest from the next branch
/v1.0.0/       Tagged release
```

GitHub Pages must be configured to deploy from the `gh-pages` branch (root).

## Reproducibility Break Protocol

This project guarantees that identical seeds produce identical results across
all supported platforms (Linux/GCC, macOS/Clang, Windows/MinGW). Golden files
in `golden/` are verified in CI on every platform.

If a code change intentionally alters model outputs for the same seed:

1. Open an issue or PR describing **why** the change is necessary
2. Regenerate golden files: `make golden-regen`
3. Verify all platforms pass: CI must be green on all three OS targets
4. Document the break in the PR description
5. Tag the release with a minor version bump

Implementation constraints that preserve reproducibility:
- **RNG**: pcg32 only (`stats::RNG`). Never `std::mt19937`.
- **Shuffling**: `stats::Uniform::distinct()` only. Never `std::shuffle`.
- **Sorting**: use `std::stable_sort` where element order affects downstream results. `std::sort` is not guaranteed to be stable and can produce different orderings of equal elements across platforms.
- **R seeds**: generated in R, passed as integers to C++.

## Versioning

The project follows [Semantic Versioning](https://semver.org/) with a single source of truth: the `VERSION` file at the repository root.

- **MAJOR** — breaking API changes (C++ public API, R/Python interface changes that break user code)
- **MINOR** — new features, new model types, new parameters
- **PATCH** — bug fixes, performance improvements, documentation

The `VERSION` file contains `MAJOR.MINOR.PATCH` (e.g., `0.1.0`). All components share the same version: CMake reads it for the C++ core and CLI, and `make r-prepare` updates the R package DESCRIPTION. Git tags use the format `v0.1.0`.

### Changelog

`CHANGELOG.md` at the repository root tracks all changes. It uses the format expected by R's `utils::news()` (`# ppforest2 X.Y.Z` headings) and is copied as `NEWS.md` into the R package during `make r-prepare`.

### How to release

1. Update the `VERSION` file with the new version number
2. Add a section to `CHANGELOG.md` for the new version
3. Run `make r-prepare` (or `make r-build`) — DESCRIPTION version is updated from the `VERSION` file
4. Commit, tag (`v0.1.0`), push

### Reverting a release

If a release was published incorrectly, you can remove its git tag:

```
make release-revert                    # Revert the current VERSION's release
make release-revert RELEASE_TAG=v0.1.0 # Revert a specific release
```

This deletes the git tag locally and from the remote. If a GitHub Release exists for the tag, it must be deleted manually from the repository's releases page. The command does not revert commits — amend the `VERSION` file and `CHANGELOG.md` as needed after reverting.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
