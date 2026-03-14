cmake_minimum_required(VERSION 3.20)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../VERSION" PPFOREST2_VERSION)
string(STRIP "${PPFOREST2_VERSION}" PPFOREST2_VERSION)
project(ppforest2 VERSION ${PPFOREST2_VERSION})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Platform detection for golden files
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(PPFOREST2_OS "linux")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(PPFOREST2_OS "macos")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  set(PPFOREST2_OS "windows")
else()
  set(PPFOREST2_OS "unknown")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(PPFOREST2_COMPILER "gcc")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(PPFOREST2_COMPILER "clang")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  set(PPFOREST2_COMPILER "msvc")
else()
  set(PPFOREST2_COMPILER "unknown")
endif()

set(PPFOREST2_PLATFORM "${PPFOREST2_OS}-${PPFOREST2_COMPILER}")

option(PPFOREST2_DOUBLE_PRECISION "Use double precision for feature data" OFF)
option(PPFOREST2_CORE_ONLY "Build only the core library, skip CLI, tests, and benchmarks" OFF)
option(PPFOREST2_NATIVE_ARCH "Use -march=native for CPU-specific optimization" ON)
if(PPFOREST2_DOUBLE_PRECISION)
  add_compile_definitions(PPFOREST2_DOUBLE_PRECISION)
endif()

add_compile_definitions(EIGEN_DONT_PARALLELIZE)
add_compile_definitions(EIGEN_NO_AUTOMATIC_RESIZING)

if(WIN32)
  # Disable vectorization to prevent alignment issues on Windows
  add_compile_definitions(EIGEN_DONT_VECTORIZE)
endif()

if(PPFOREST2_NATIVE_ARCH AND NOT CMAKE_BUILD_TYPE MATCHES Debug)
  add_compile_options(-march=native)
endif()

if(CMAKE_BUILD_TYPE MATCHES Debug)
  add_compile_options(-O0)
  add_compile_options(-g)
  add_compile_definitions(_GLIBCXX_DEBUG)
  add_compile_definitions(_GLIBCXX_ASSERTIONS)
  add_compile_definitions(EIGEN_INITIALIZE_MATRICES_BY_NAN)
  add_compile_definitions(EIGEN_INTERNAL_DEBUGGING)
  add_compile_definitions(EIGEN_RUNTIME_NO_MALLOC)
  add_compile_options(-fno-omit-frame-pointer)

  add_link_options(--coverage)

  if(UNIX AND NOT APPLE)
    add_compile_options(-fsanitize=address,undefined)
    add_link_options(-fsanitize=address,undefined)
  endif()

  add_link_options(--coverage)

  if(WIN32)
    add_compile_options(-Wa,-mbig-obj)
  endif()
endif()

if(CMAKE_BUILD_TYPE MATCHES Release)
  add_compile_options(-O3)
  add_compile_definitions(NDEBUG)
  add_compile_definitions(EIGEN_NO_DEBUG)
endif()

if(WIN32)
  add_link_options(-static)
endif()

# Add search hints
list(APPEND CMAKE_PREFIX_PATH
  /usr/local/opt/libomp
  /opt/homebrew/opt/libomp)

# Find OpenMP
find_package(OpenMP)

# Find Jemalloc using pkg-config
include(FindPkgConfig)
pkg_check_modules(Jemalloc IMPORTED_TARGET jemalloc)

if(NOT OpenMP_FOUND)
  message(WARNING "Could NOT find OpenMP. Parallelization will not be available.")
endif()

include(Dependencies.cmake)

# ---------------------------------------------------------------------------
# Common compile interface for all ppforest2 OBJECT libraries
# ---------------------------------------------------------------------------
add_library(ppforest2-compile INTERFACE)

target_include_directories(ppforest2-compile INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(ppforest2-compile INTERFACE
  Eigen3::Eigen
  pcg_lib)

if(OpenMP_FOUND)
  target_link_libraries(ppforest2-compile INTERFACE OpenMP::OpenMP_CXX)
endif()

# ---------------------------------------------------------------------------
# OBJECT libraries (one per subfolder)
# ---------------------------------------------------------------------------
add_subdirectory(src/models)
add_subdirectory(src/stats)
add_subdirectory(src/serialization)
add_subdirectory(src/utils)
add_subdirectory(src/io)

# ---------------------------------------------------------------------------
# Core library (composed from OBJECT libraries)
# ---------------------------------------------------------------------------
add_library(ppforest2-core
  $<TARGET_OBJECTS:ppforest2-models>
  $<TARGET_OBJECTS:ppforest2-stats>
  $<TARGET_OBJECTS:ppforest2-serialization>
  $<TARGET_OBJECTS:ppforest2-utils>
  $<TARGET_OBJECTS:ppforest2-io>)

target_link_libraries(ppforest2-core PUBLIC ppforest2-compile nlohmann_json::nlohmann_json)

if(Jemalloc_FOUND)
  message(STATUS "Jemalloc found: ${Jemalloc_INCLUDE_DIRS}")
  target_link_libraries(ppforest2-core PRIVATE PkgConfig::Jemalloc)
else()
  message(STATUS "Jemalloc not found. Multi-threaded allocation performance may be suboptimal.")
endif()

if(NOT PPFOREST2_CORE_ONLY)
  # ---------------------------------------------------------------------------
  # CLI
  # ---------------------------------------------------------------------------
  add_subdirectory(src/cli)

  add_executable(ppforest2 src/cli/CLI.cpp $<TARGET_OBJECTS:ppforest2-cli>)
  target_include_directories(ppforest2 PRIVATE ${csv_SOURCE_DIR}/include)
  target_link_libraries(ppforest2 ppforest2-core ppforest2-serialization nlohmann_json::nlohmann_json csv CLI11::CLI11 fmt::fmt)
  target_compile_definitions(ppforest2 PRIVATE PPFOREST2_VERSION="${PROJECT_VERSION}")

  # ---------------------------------------------------------------------------
  # Tests
  # ---------------------------------------------------------------------------
  enable_testing()

  set(PPFOREST2_TEST_SRC
    src/test.cpp
    src/models/Tree.test.cpp
    src/models/Forest.test.cpp
    src/models/PPStrategy.test.cpp
    src/models/VariableImportance.test.cpp
    src/models/Visualization.test.cpp
    src/stats/Stats.test.cpp
    src/stats/GroupPartition.test.cpp
    src/stats/ConfusionMatrix.test.cpp
    src/stats/Uniform.test.cpp
    src/stats/Normal.test.cpp
    src/stats/Simulation.test.cpp
    src/serialization/Json.test.cpp
    src/utils/Math.test.cpp
    src/utils/Invariant.test.cpp
    src/utils/System.test.cpp
    src/cli/CLIOptions.test.cpp
    src/cli/Train.integration.test.cpp
    src/cli/Predict.integration.test.cpp
    src/cli/Evaluate.integration.test.cpp
    src/cli/Summarize.integration.test.cpp
    src/cli/Benchmark.integration.test.cpp
    src/cli/Reproducibility.integration.test.cpp
    src/cli/CLI.integration.test.cpp
    src/golden/Reproducibility.test.cpp
    src/golden/Threading.test.cpp
    src/cli/VarsSpec.test.cpp
    src/cli/Benchmark.test.cpp
    src/cli/BenchmarkReport.test.cpp
    src/io/IO.test.cpp
    src/io/Presentation.test.cpp)

  add_executable(ppforest2-test ${PPFOREST2_TEST_SRC} $<TARGET_OBJECTS:ppforest2-cli>)

  target_link_libraries(ppforest2-test
    ppforest2-core
    ppforest2-serialization
    nlohmann_json::nlohmann_json
    gtest
    CLI11::CLI11
    csv
    fmt::fmt)

  target_include_directories(ppforest2-test PRIVATE
    ${csv_SOURCE_DIR}/include
    ${gtest_SOURCE_DIR}/include
    ${gtest_SOURCE_DIR})

  target_compile_definitions(ppforest2-test PRIVATE
    PPFOREST2_BINARY_PATH="${CMAKE_BINARY_DIR}/ppforest2"
    PPFOREST2_DATA_DIR="${CMAKE_SOURCE_DIR}/../data"
    PPFOREST2_GOLDEN_DIR="${CMAKE_SOURCE_DIR}/../golden"
    PPFOREST2_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/../fixtures"
    PPFOREST2_BENCH_SCENARIOS="${CMAKE_SOURCE_DIR}/../bench/default-scenarios.json"
    PPFOREST2_PLATFORM="${PPFOREST2_PLATFORM}"
    PPFOREST2_VERSION="${PROJECT_VERSION}")

  add_dependencies(ppforest2-test ppforest2)

  include(GoogleTest)
  gtest_discover_tests(ppforest2-test DISCOVERY_MODE PRE_TEST)

  # ---------------------------------------------------------------------------
  # Golden file generator
  # ---------------------------------------------------------------------------
  add_executable(ppforest2-golden-gen src/golden/GoldenGen.cpp src/cli/Metrics.cpp)

  target_link_libraries(ppforest2-golden-gen
    ppforest2-core
    ppforest2-serialization
    nlohmann_json::nlohmann_json
    csv
    fmt::fmt)

  target_include_directories(ppforest2-golden-gen PRIVATE
    ${csv_SOURCE_DIR}/include)

  target_compile_definitions(ppforest2-golden-gen PRIVATE
    PPFOREST2_DATA_DIR="${CMAKE_SOURCE_DIR}/../data"
    PPFOREST2_GOLDEN_DIR="${CMAKE_SOURCE_DIR}/../golden"
    PPFOREST2_PLATFORM="${PPFOREST2_PLATFORM}")
endif()
