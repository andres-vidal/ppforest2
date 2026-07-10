# ppforest2

[![C++ Tests](https://github.com/andres-vidal/ppforest2/actions/workflows/run-test.yml/badge.svg)](https://github.com/andres-vidal/ppforest2/actions/workflows/run-test.yml)
[![R Tests](https://github.com/andres-vidal/ppforest2/actions/workflows/run-r-test.yml/badge.svg)](https://github.com/andres-vidal/ppforest2/actions/workflows/run-r-test.yml)
[![R Package Check](https://github.com/andres-vidal/ppforest2/actions/workflows/run-r-check.yml/badge.svg)](https://github.com/andres-vidal/ppforest2/actions/workflows/run-r-check.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/andres-vidal/aafefce6b546eeb2f678ca607a950941/raw/ppforest2-coverage.json)](https://github.com/andres-vidal/ppforest2/actions/workflows/run-coverage.yml)

**ppforest2** is a fast, memory-efficient implementation of
[Projection Pursuit Random Forests](https://www.tandfonline.com/doi/full/10.1080/10618600.2020.1870480),
built on
[Projection Pursuit (oblique) Decision Trees](https://projecteuclid.org/journals/electronic-journal-of-statistics/volume-7/issue-none/PPtree-Projection-pursuit-classification-tree/10.1214/13-EJS810.full).
By learning linear projections at each split, the model captures complex structure that axis-aligned trees often miss, without sacrificing interpretability or scalability.

The project provides a high-performance C++ core with interfaces for **R** and a command-line interface (CLI), with **Python** bindings planned. In the R ecosystem, it is intended as a modern successor to
[`PPforest`](https://cran.r-project.org/web/packages/PPforest/index.html),
offering the same statistical foundations with significantly improved computational performance.

Developed by [Andrés Vidal](https://andresvidal.dev) as a Bachelor's thesis project in Statistics at **Universidad de la República (Uruguay)**.

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

# Regression (experimental) — last CSV column must be a continuous numeric response
ppforest2 train --data regression.csv --mode regression --trees 100 --save reg_model.json

# Explicit strategy selection (equivalent to the above)
ppforest2 train --data data.csv --trees 100 --pp-strategy pda:lambda=0.5 --save model.json

# Predict on new data using a saved model
ppforest2 predict --model model.json --data test.csv

# Evaluate with smart convergence (default)
ppforest2 evaluate --data data.csv --trees 100 --train-ratio 0.7

# Evaluate with fixed iterations (disables convergence)
ppforest2 evaluate --data data.csv --trees 100 -i 10 --train-ratio 0.7

# Evaluate on simulated data (1000 rows, 10 features, 3 groups)
ppforest2 evaluate --simulate 1000x10x3 --trees 50

# Run performance benchmarks across scenarios
ppforest2 benchmark -s bench/default-scenarios-classification.json

# Serve predictions over HTTP from a saved model (localhost:8080 by default)
ppforest2 serve --model model.json
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
model <- pptr(Species ~ ., data = iris)
predict(model, iris)

# Random forest with 50 trees
forest <- pprf(Species ~ ., data = iris, size = 50)
predict(forest, iris)
predict(forest, iris, type = "prob")   # vote proportions
summary(forest)                        # variable importance & model info

# Explicit strategy selection (equivalent to lambda + n_vars shortcuts)
forest <- pprf(Species ~ ., data = iris, size = 50, pp = pp_pda(0.5), vars = vars_uniform(count = 2))

# Regression (mode auto-detected from numeric y) — experimental, untested in production
reg_model <- pprf(mpg ~ ., data = mtcars, size = 50)
predict(reg_model, mtcars)              # numeric predictions (MSE-based OOB)
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
model2 <- pptr(x = iris[, 1:2], y = iris$Species)
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
fit <- spec %>% fit(Species ~ ., data = iris)
predict(fit, iris, type = "prob")
```

Tunable with `tune()`: `trees`, `mtry` (or `mtry_prop` to subsample features by proportion), and `penalty`.

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

| Flag                     | Default       | Description                                                        |
|--------------------------|---------------|--------------------------------------------------------------------|
| `-d, --data <file>`      | *(required)*  | CSV training data                                                  |
| `-t, --trees <N>`        | `100`         | Number of trees (`0` for a single tree)                            |
| `-l, --lambda <X>`       | `0.5`         | PDA penalty; `0` = LDA, `(0,1]` = PDA                              |
| `-r, --seed <N>`         | *(random)*    | Random seed for reproducibility                                    |
| `-v, --vars <spec>`      | `0.5`         | Features per split (see [Variable selection](#variable-selection)) |
| `--threads <N>`          | *(all cores)* | Number of OpenMP threads                                           |
| `--max-retries <N>`      | `3`           | Max retries for degenerate trees                                   |
| `--pp-strategy <spec>`        | `pda:lambda=0.5` | PP strategy (e.g. `pda:lambda=0.5`); excludes `--lambda`       |
| `--vars-strategy <spec>`      | `uniform`     | Variable selection strategy (e.g. `all`, `uniform:count=3`); excludes `--vars` |
| `--cutpoint-strategy <spec>`  | `mean_of_means` | Cutpoint strategy (e.g. `mean_of_means`)                       |
| `--stop-strategy <spec>`      | `pure_node`   | Stop rule (e.g. `pure_node`)                                      |
| `--binarize-strategy <spec>`  | `largest_gap` | Binarization strategy (e.g. `largest_gap`)                         |
| `--grouping <spec>`           | `by_label`    | Grouping strategy (e.g. `by_label`)                                |
| `-s, --save <file>`      | `model.json`  | Output model path (`.json` added if missing)                       |
| `--no-save`              | —             | Skip saving the model                                              |
| `--no-metrics`           | —             | Skip variable importance computation                               |

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

All model parameters (`--trees`, `--lambda`, `--seed`, `--vars`, `--threads`, `--max-retries`) and strategy flags (`--pp-strategy`, `--vars-strategy`, `--cutpoint-strategy`, `--stop-strategy`, `--binarize-strategy`, `--grouping`) are also available.

### `benchmark` — Multi-Scenario Benchmarks

Run a suite of evaluation scenarios and report results in a table. Each scenario runs as a separate subprocess for accurate per-scenario memory measurement.

```bash
ppforest2 benchmark -s bench/default-scenarios-classification.json
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

### `serve` — HTTP Inference Server

Load a saved model into memory and expose `GET /` (model summary — HTML for browsers, JSON for API clients), `GET /health`, and `POST /predict` over HTTP. The model path is set once at startup; clients never pass paths in requests.

```bash
# Start a server on localhost:8080
ppforest2 serve --model model.json

# Bind elsewhere or raise the body cap
ppforest2 serve --model model.json --host 127.0.0.1 --port 9000 --max-body-bytes 4194304
```

| Flag                     | Default       | Description                                       |
|--------------------------|---------------|---------------------------------------------------|
| `-M, --model <file>`     | *(required)*  | Saved model JSON                                  |
| `--host <addr>`          | `127.0.0.1`   | Bind address. `0.0.0.0` exposes externally (prints a warning) |
| `--port <n>`             | `8080`        | TCP port (1–65535)                                |
| `--max-body-bytes <n>`   | `1048576`     | Maximum `POST /predict` body size                 |

**Endpoints**

Content type is chosen by the request's `Accept` header — browsers (`text/html`) get the dashboard, everything else (`*/*`, explicit `application/json`) gets JSON.

- `GET /` — model summary. JSON returns `meta`, `config`, `training_metrics`, `oob_metrics`, `variable_importance` (the heavy `model` tree is stripped). HTML returns the dashboard.
- `GET /predict` — predict landing page (HTML only). Without a query string: upload form. With `?id=<hex>`: a previously-computed result from the in-memory cache, with a download button.
- `POST /predict` — body is feature-only CSV (header + rows). JSON returns `{predictions, [proportions], [metrics], id, url}`. HTML returns the predictions table (plus metrics and a colour-coded confusion matrix when the request CSV included a response column). `Content-Location` and the body's `url` field point to the canonical `/predict?id=<hex>` URL so a single result is refreshable and shareable.
- `GET /health` — liveness probe. Returns `{"status":"ok","version":"X.Y.Z"}`.

```bash
# Browser → http://127.0.0.1:8080/ shows the model dashboard.
curl -s -H 'Accept: application/json' http://127.0.0.1:8080/ | jq 'keys'
curl -s http://127.0.0.1:8080/health
curl -s -X POST -H 'Content-Type: text/csv' \
     --data-binary $'Sepal.Length,Sepal.Width,Petal.Length,Petal.Width\n5.1,3.5,1.4,0.2\n6.7,3.0,5.2,2.3' \
     http://127.0.0.1:8080/predict | jq
```

**Request format**: header columns map to the model's training-time feature names; column order may differ from the training CSV. Missing columns produce a `400` with a list of offending names. Extra columns (e.g. a response column carried over from the training CSV) are silently ignored for prediction — but if the rightmost extra parses as the model's response type (classification labels in `meta.groups` or numeric values for regression), it's used as ground truth and the response gains a `metrics` field with accuracy/error-rate or MAE/MSE/R². Categorical features must be encoded by the caller in the same form they had during training.

**Deployment notes**: the default bind is `127.0.0.1` because the server is intentionally minimal — no TLS, no auth, no rate limiting. Those belong in front, not in here. The prediction cache is a bounded in-memory LRU (~32 entries) cleared on restart, so `?id=<hex>` URLs survive a page refresh but not a deploy. For per-user persistence or richer access control, hit the JSON endpoints from your own application.

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

- **Numeric precision** — The C++ core uses single-precision (`float`) arithmetic for all feature data. This is sufficient for classification and reduces memory usage (benchmarks show double precision costs 10–90% more time and up to 45% more memory with no accuracy benefit for classification). If a future strategy needs higher precision internally (e.g., regression loss computation), it can cast to `double` within its own scope without affecting the rest of the pipeline.

- **Regression support** — Available as of v0.1.0 (experimental, untested in production). Enabled via `--mode regression` on the CLI or automatically detected from numeric `y` in R. Uses a `ByCutpoint` grouping strategy that partitions observations by the cutpoint in projected space, then re-clusters each child via a median split of the continuous response for the next PDA step. Leaves predict the mean response. Forests aggregate by averaging tree predictions (rather than majority vote) and bootstrap via uniform sampling (rather than stratified). OOB error is reported as MSE.

- **Python bindings** — Planned.

### Design patterns

The C++ core uses two design patterns to keep the algorithm extensible without heavily modifying existing code:

- **Strategy** — Seven strategy families control each step of node training: projection pursuit (`ppforest2::pp::ProjectionPursuit`), variable selection (`ppforest2::vars::VariableSelection`), cutpoint (`ppforest2::cutpoint::Cutpoint`), stopping rule (`ppforest2::stop::StopRule`), binarization (`ppforest2::binarize::Binarization`), grouping (`ppforest2::grouping::Grouping`), and leaf assignment (`ppforest2::leaf::LeafStrategy`). Concrete implementations (e.g. `pp::PDA`, `vars::Uniform`, `binarize::LargestGap`, `leaf::MajorityVote`) are composed at runtime via `TrainingSpec`, a single concrete class that holds the seven strategy objects (via `shared_ptr`) together with forest-level parameters (size, seed, threads, max retries). All strategies share a uniform `(NodeContext&, RNG&)` interface — each reads what it needs from the mutable context and writes its results back. New optimization criteria, stopping rules, or partition methods can be added without changing the tree-building logic — just implement the interface and pass it to `TrainingSpec`. Each strategy implements `to_json()` for serialization and is auto-registered via a CRTP macro.

- **Visitor** — `TreeNode::Visitor` dispatches over the two node types (internal `TreeBranch` and leaf `TreeLeaf`) and `Model::Visitor` dispatches over `Tree` and `Forest`. This avoids `dynamic_cast` and keeps traversal logic (serialization, visualization layout, variable importance) decoupled from the model types themselves.

## Prerequisites

|              | Linux            | macOS           | Windows                              |
|--------------|------------------|-----------------|--------------------------------------|
| **C++ core** | `cmake` >= 3.20, `make`, `gcc` | `cmake` >= 3.20, `make`, `clang` | `cmake` >= 3.20, `make`, MinGW `gcc` |
| **R package**| `R` >= 3.5       | `R` >= 3.5      | `R` >= 3.5, `Rtools`                |
| **OpenMP** (optional) | Usually included with `gcc` | `brew install libomp` | Usually included with MinGW |
| **Coverage** (optional) | `lcov` >= 2  | `brew install lcov` | —                                    |
| **R docs**   | TeX distribution with `pdflatex` | TeX distribution with `pdflatex` | TeX distribution with `pdflatex` |

For the R package, the C++ compiler must match the one R was built with (`gcc` on Linux/Windows, `clang` on macOS). OpenMP is optional but recommended for multi-threaded forest training; without it, forests train on a single thread.

## Building and Testing

### C++ core

```bash
make build              # Release build (C++ core + CLI + tests)
make test               # Build and run C++ tests (GoogleTest)
make build-debug        # Debug build with AddressSanitizer
make test-debug         # Run debug tests
make coverage           # Build with coverage, run tests, generate report and HTML
make clean              # Remove all build artifacts (.build/, .debug/, .coverage/, .r-build/)
```

The release build produces the `ppforest2` CLI binary and the `ppforest2-test` test runner in `.build/`. The debug build enables AddressSanitizer (on Linux) and runtime assertions.

### R package

```bash
make r-install-deps     # Install R package dependencies via pak
make r-build            # Prepare source and run R CMD build (produces tarball)
make r-test             # Fast: install + run R tests only (devtools)
make r-check            # Build and run R CMD check on the tarball
make r-check-cran       # Same as r-check with --as-cran for CRAN submission
make r-install          # Build and install the package locally
make r-document         # Regenerate Roxygen man pages
make r-clean            # Remove R compilation byproducts
```

### Development tools

Dev tools (clang-format, clang-tidy, cppcheck) are installed via pip with pinned versions in `requirements-dev.txt`. Doxygen is built from source separately (only needed for documentation).

```bash
make install-tools      # Install dev tools (pip + doxygen)
make format             # Format C++ code (clang-format)
make format-dry         # Check formatting without applying changes
make analyze            # Run static analysis (cppcheck)
make tidy               # Run static analysis (clang-tidy)
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
make benchmark                     # Run both classification + regression suites, print tables
make benchmark-save                # Save results to bench/results-classification.{json,csv} + bench/results-regression.{json,csv}
make benchmark-compare             # Compare both suites against their saved baselines
make benchmark-vs REF=main         # Compare current branch against another ref (branch/tag/commit)
```

### CLI Usage

```bash
# Run scenarios from a JSON file
ppforest2 benchmark -s bench/default-scenarios-classification.json

# Save results as JSON and CSV
ppforest2 benchmark -s bench/default-scenarios-classification.json -o results.json --csv results.csv

# Compare against a baseline
ppforest2 benchmark -s bench/default-scenarios-classification.json -b baseline.json

# Override iteration count (forces fixed mode)
ppforest2 benchmark -s bench/default-scenarios-classification.json -i 5
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
    "seed": 0, "warmup": 2,
    "convergence": { "cv": 0.05, "max": 200 }
  },
  "scenarios": [
    { "name": "small-forest",  "n": 200,  "p": 5,  "g": 2, "trees": 50 },
    { "name": "medium-forest", "n": 1000, "p": 20, "g": 3 },
    { "name": "fixed-5",       "n": 1000, "p": 20, "g": 3, "iterations": 5 },
    { "name": "data-iris",     "data": "data/classification/iris.csv", "trees": 50 }
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
make r-test             # Fast: install + run R tests only
make r-check            # Run R CMD check on the built tarball
make r-install          # Run R CMD INSTALL on the built tarball
make r-document         # Regenerate Roxygen man pages
make r-clean            # Remove compilation byproducts
```

> **Important:** Always use `make r-build` before checking or installing. This target copies the C++ core source into the R package's `src/core/` so it can be compiled on install.

### Build Process

The R package wraps the C++ core via Rcpp. Because the core lives outside the R package directory, the build process assembles a self-contained source tarball. The C++ core is compiled **directly through `Makevars`** using R's own compiler — no CMake, no network access, and no pre-built static libraries. This keeps the package CRAN-installable: everything it needs ships in the tarball.

The only external C++ dependencies at R-build time are Eigen (provided by `RcppEigen` via `LinkingTo`) and the header-only nlohmann/json and pcg libraries, which are vendored under `bindings/R/inst/include/`. The `fmt` and `csv-parser` dependencies are only used by the CLI/io layer, which the R package does not compile.

#### Tarball pipeline (`make r-check`)

1. **`r-prepare`** — Stages the core sources the R package compiles into `src/core/` (dropping the `cli/`, `io/`, `golden/` and test translation units, which need `fmt`/`csv-parser`/GoogleTest), copies the changelog to `NEWS.md`, and copies golden files into `inst/golden/`. The vendored json/pcg headers are already committed under `inst/include/`, so nothing is downloaded.

2. **`r-build`** — Regenerates `RcppExports.cpp`/`RcppExports.R` via `Rcpp::compileAttributes()`, then runs `R CMD build` to produce a source tarball.

3. **`configure` / `configure.win`** — During `R CMD INSTALL` (or `R CMD check`), the configure script:
   - Stages the core sources if needed — from the bundled `src/core/` in a **tarball**, or by copying `../../core/` in the **monorepo** (`devtools::load_all()` / `install_github`).
   - Generates the `OBJECTS` list (every needed core `.cpp`) and writes `src/Makevars` from `Makevars.in`.
   - Detects OpenMP support (needed only for the macOS special case; elsewhere R's `$(SHLIB_OPENMP_CXXFLAGS)` suffices) and falls back to a single-threaded build if unavailable.

   Compiling every strategy object straight into the shared object means the self-registering strategies cannot be dead-stripped, so no whole-archive link trickery is required.

#### Development workflow (`devtools::load_all()`)

`devtools::load_all()` runs `configure`, which stages `../../core/` into `src/core/` and compiles it through `Makevars` — the same path used for a tarball install.

```r
devtools::load_all("bindings/R")   # edit C++ -> reload -> test
devtools::test("bindings/R")        # run testthat suite
```

### How `install_github` Works

`install_github` requires `build = FALSE` so that `R CMD INSTALL` runs directly on the source directory within the cloned monorepo (without `build = FALSE`, `R CMD build` creates an intermediate tarball that loses the monorepo context):

```r
devtools::install_github("andres-vidal/ppforest2", subdir = "bindings/R", build = FALSE)
```

The configure script detects `../../core/`, stages it into `src/core/`, and compiles it through `Makevars`.

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

## Known limitations

### LDA fidelity at `lambda = 0` with more variables than observations

ppforest2 targets fidelity to `PPforest`, but diverges in one corner: pure LDA
(`lambda = 0`) with more variables than effective observations — high-dimensional
data with all variables active (`p_vars = 1`), or a duplicate-heavy bootstrap
sample whose unique-row count falls below the feature count. There the
within-group scatter `W` is rank-deficient, so the metric matrix `W + B` of the
LDA generalized eigenproblem is singular or near-singular.

ppforest2 solves that eigenproblem with a symmetric-definite eigensolver
(`Eigen::GeneralizedSelfAdjointEigenSolver`), which requires `W + B` positive
definite. In this corner it either degenerates to a majority-class leaf (with a
warning) or, when `W + B` is only near-singular, returns an unstable direction
that overfits — whereas `PPforest` recovers a usable direction via an explicit
(pseudo-)inverse (`arma::inv`; `PPtreeViz` uses `MASS::ginv`) plus a general
eigensolver. On the high-dimensional gene-expression datasets (`leukemia`,
`lymphoma`, `NCI60`) this surfaces as markedly higher error for ppforest2 at
`lambda = 0, p_vars = 1` only.

This is a fidelity gap, not a correctness bug: LDA is genuinely ill-posed when
`W` is singular. Any regularization — `lambda > 0` (PDA) or `p_vars < 1` —
restores a well-conditioned problem and closes the gap. ppforest2 therefore
defaults to `lambda = 0.5` (PDA), so this corner is only reached by explicitly
requesting pure LDA (`lambda = 0`). Root cause is documented at the eigensolver in
`core/src/models/strategies/pp/PDA.cpp`.

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
