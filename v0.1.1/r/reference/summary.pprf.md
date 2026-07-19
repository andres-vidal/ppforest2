# Summary of a pprf forest (shared header + VI).

Summary of a pprf forest (shared header + VI).

## Usage

``` r
# S3 method for class 'pprf'
summary(object, ...)
```

## Arguments

- object:

  A `pprf` model.

- ...:

  Unused.

## Value

Invisibly returns the input `pprf` model `object` (unchanged). Called
for its side effect of printing a detailed summary – the training
specification, data summary, and variable-importance table (plus, for
classification, the training/OOB confusion matrices) – to the console.
