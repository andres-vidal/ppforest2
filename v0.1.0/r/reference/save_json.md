# Save a model to a JSON file.

Serializes a `pptr` or `pprf` model to JSON format compatible with the
C++ CLI. The JSON includes the model structure, group labels, training
parameters, and optionally variable importance metrics.

## Usage

``` r
save_json(model, path, ...)

# S3 method for class 'ppmodel'
save_json(model, path, include_metrics = TRUE, ...)
```

## Arguments

- model:

  A `pptr` or `pprf` model (inheriting from `ppmodel`).

- path:

  File path to write the JSON to.

- ...:

  Additional arguments (currently unused).

- include_metrics:

  If `TRUE` (default), include variable importance and (for forests) OOB
  error in the output. Set to `FALSE` to save only the model structure
  and metadata.

## See also

[`load_json`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/load_json.md),
[`pptr`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/pptr.md),
[`pprf`](https://andres-vidal.github.io/ppforest2/v0.1.0/r/reference/pprf.md)

## Examples

``` r
model <- pptr(Species ~ ., data = iris, seed = 0)
path <- tempfile(fileext = ".json")
save_json(model, path)
```
