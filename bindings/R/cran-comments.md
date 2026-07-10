## Resubmission

This is a resubmission. In response to the review, I have added \value tags to
the four exported methods whose .Rd files were missing them, documenting both
the returned value (structure/class) and the function's side effect:

* print.pprf, print.pptr, summary.pprf: each invisibly returns the input model
  (unchanged) and is called for its side effect of printing to the console.
* save_json: invisibly returns the written file path (a length-one character
  string) and is called for its side effect of writing the model to a JSON file.

## Submission

This is a new submission of ppforest2, an R interface to a portable C++
implementation of projection-pursuit oblique decision trees and random forests.

## Test environments

* local macOS (R release), R CMD check --as-cran
* GitHub Actions: Ubuntu, macOS, and Windows (R release), R CMD check --as-cran
* R-devel (Linux, r-hub ubuntu-next container), R CMD check --as-cran
* AddressSanitizer + UndefinedBehaviorSanitizer, clang and gcc (r-hub containers)
* valgrind (r-hub container)

The package compiles a C++ core, so it was additionally checked under the
sanitizer and valgrind toolchains; no memory or undefined-behaviour issues
were reported.

## R CMD check results

0 errors | 0 warnings | 1 note

* checking CRAN incoming feasibility ... NOTE
  Maintainer: 'Andrés Vidal <andres@andresvidal.dev>'
  New submission

  This is the expected note for a first-time submission.

  The incoming check also flags "da" as possibly misspelled in the
  Description; this is part of the author name "da Silva" (Natalia da Silva)
  and is spelled correctly.

## Notes for the reviewer

* The package bundles a small amount of third-party C++ header code
  (nlohmann/json and the PCG random number generator) under `inst/include/`,
  used by the C++ core. The only modification to the vendored nlohmann/json
  headers is the removal of their `#pragma GCC/clang diagnostic ignored`
  lines, so the library suppresses no compiler diagnostics; the headers are
  otherwise upstream and functionally unchanged. Eigen is obtained from
  RcppEigen via LinkingTo. No code is downloaded at build or install time;
  the package builds entirely from the sources in the tarball and does not
  require CMake.

* OpenMP is used for optional multi-threaded forest training and is guarded by
  `#ifdef _OPENMP`; the package builds and runs correctly without it.

* All random number generation in the C++ core is seeded explicitly from R via
  `set.seed()`, so results are reproducible across platforms.
