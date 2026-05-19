include(FetchContent)

# Eigen
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.1
)
FetchContent_MakeAvailable(eigen)

# nlohmann_json
FetchContent_Declare(
  json
  URL
  https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(json)

# PCG Random Number Generation
FetchContent_Declare(
  pcg
  GIT_REPOSITORY https://github.com/imneme/pcg-cpp.git
  GIT_TAG v0.98.1
)
FetchContent_MakeAvailable(pcg)

# Create interface library for PCG
add_library(pcg_lib INTERFACE)
target_include_directories(pcg_lib INTERFACE ${pcg_SOURCE_DIR}/include)

# Fetch csv-parser
FetchContent_Declare(
  csv
  GIT_REPOSITORY https://github.com/vincentlaucsb/csv-parser.git
  GIT_SHALLOW TRUE
  GIT_TAG 2.5.1
)
FetchContent_MakeAvailable(csv)

# fmt for formatted output and color
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 12.1.0
)
FetchContent_MakeAvailable(fmt)

if(NOT PPFOREST2_CORE_ONLY)
  # CLI11 for command line parsing
  FetchContent_Declare(
    cli11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG v2.6.1
  )
  FetchContent_MakeAvailable(cli11)

  # Google Test
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.17.0
    SYSTEM
  )
  FetchContent_MakeAvailable(googletest)

  # inja for HTML templating (Jinja2-like, header-only) — backs the
  # serve dashboard's templates. Depends on nlohmann_json which we already
  # fetch unconditionally above.
  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
  set(BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
  set(INJA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(INJA_USE_EMBEDDED_JSON OFF CACHE BOOL "" FORCE)
  set(INJA_INSTALL OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    inja
    GIT_REPOSITORY https://github.com/pantor/inja.git
    GIT_TAG v3.4.0
  )
  FetchContent_MakeAvailable(inja)

  # cpp-httplib for `serve` subcommand.
  # Disable optional codecs and OpenSSL — we only need plain HTTP so callers
  # don't pick up Brotli/zlib/OpenSSL link dependencies just by linking
  # `ppforest2-cli`. TLS and gzip are out of scope (caller responsibility,
  # behind a reverse proxy if exposed).
  set(HTTPLIB_USE_OPENSSL_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
  set(HTTPLIB_USE_ZLIB_IF_AVAILABLE    OFF CACHE BOOL "" FORCE)
  set(HTTPLIB_USE_BROTLI_IF_AVAILABLE  OFF CACHE BOOL "" FORCE)
  set(HTTPLIB_USE_CERTS_FROM_MACOSX_KEYCHAIN OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG v0.18.5
  )
  FetchContent_MakeAvailable(httplib)
endif()
