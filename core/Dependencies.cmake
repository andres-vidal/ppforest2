include(FetchContent)

# Eigen
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 5.0.1
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
  )
  FetchContent_MakeAvailable(googletest)
endif()
