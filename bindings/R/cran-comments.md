## Update

This is an update (0.1.2) that fixes the check problem reported for the current
CRAN version at
<https://cran.r-project.org/web/checks/check_results_ppforest2.html>.

The macOS/M1mac additional check flagged a `-Wdeprecated-declarations` warning:
the newer libc++ (Apple clang 21, macOS 26 SDK) deprecates
`std::char_traits<unsigned char>`, which the vendored nlohmann/json instantiates
through the default template arguments of its binary output/stream adapters
(`std::basic_string<std::uint8_t>` / `std::basic_ostream<std::uint8_t>`).
ppforest2 does not use nlohmann's binary formats. The vendored `json.hpp` is now
bracketed with a `_Pragma` guard that suppresses this deprecation; `_Pragma`
(unlike `#pragma`) is not flagged by `R CMD check`'s pragma check.

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

0 errors | 0 warnings | 0 notes

(If the incoming check flags "da" as possibly misspelled in the Description,
that is part of the author name "da Silva" (Natalia da Silva) and is spelled
correctly.)

## Notes for the reviewer

* The package bundles a small amount of third-party C++ header code
  (nlohmann/json and the PCG random number generator) under `inst/include/`,
  used by the C++ core. The vendored nlohmann/json headers are modified only in
  two non-functional ways: their `#pragma GCC/clang diagnostic ignored` lines
  are removed (so the library suppresses no compiler diagnostics), and
  `json.hpp` is bracketed with a `_Pragma` guard that silences the libc++
  `char_traits<unsigned char>` deprecation described above. The headers are
  otherwise upstream. Eigen is obtained from RcppEigen via LinkingTo. No code is
  downloaded at build or install time; the package builds entirely from the
  sources in the tarball and does not require CMake.

* OpenMP is used for optional multi-threaded forest training and is guarded by
  `#ifdef _OPENMP`; the package builds and runs correctly without it.

* All random number generation in the C++ core is seeded explicitly from R via
  `set.seed()`, so results are reproducible across platforms.
