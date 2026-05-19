cmake_minimum_required(VERSION 3.24)  # $<LINK_LIBRARY:WHOLE_ARCHIVE,...> for self-registering strategies

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

option(PPFOREST2_CORE_ONLY "Build only the core library, skip CLI, tests, and benchmarks" OFF)

add_compile_definitions(EIGEN_DONT_PARALLELIZE)
add_compile_definitions(EIGEN_NO_AUTOMATIC_RESIZING)

if(WIN32)
  # Disable vectorization to prevent alignment issues on Windows
  add_compile_definitions(EIGEN_DONT_VECTORIZE)
endif()

option(PPFOREST2_COVERAGE "Enable code coverage instrumentation" OFF)

if(CMAKE_BUILD_TYPE MATCHES Debug)
  add_compile_options(-O0)
  add_compile_options(-g)
  add_compile_definitions(_GLIBCXX_DEBUG)
  add_compile_definitions(_GLIBCXX_ASSERTIONS)
  add_compile_definitions(EIGEN_INITIALIZE_MATRICES_BY_NAN)
  add_compile_definitions(EIGEN_INTERNAL_DEBUGGING)
  add_compile_definitions(EIGEN_RUNTIME_NO_MALLOC)
  add_compile_options(-fno-omit-frame-pointer)

  if(PPFOREST2_COVERAGE)
    add_compile_options(--coverage)
    add_link_options(--coverage)
  elseif(UNIX AND NOT APPLE)
    add_compile_options(-fsanitize=address,undefined)
    add_link_options(-fsanitize=address,undefined)
  endif()

  if(WIN32)
    add_compile_options(-Wa,-mbig-obj)
  endif()
endif()

if(CMAKE_BUILD_TYPE MATCHES Release)
  add_compile_options(-O3)
  add_compile_definitions(NDEBUG)
  add_compile_definitions(EIGEN_NO_DEBUG)
endif()

# GCC 12+ runs aggressive -Wstringop-overflow= analysis that triggers
# false positives inside Eigen's IndexedView + product template chain
# (e.g. `x(indices, all).row(i).dot(projector)` ends up inlining
# multiple template levels, and GCC's diagnostic loses track of the
# actual destination buffer size, claiming a tiny region from one
# specialisation while the real allocation is dynamic). The generated
# code is correct — see Eigen issues #2375 / #2407 — only the warning
# is bogus. Clang doesn't emit these. Scope the suppression to GCC so
# we still catch real overflows on the other compilers.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12)
  add_compile_options(-Wno-stringop-overflow)
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
# Core libraries
# ---------------------------------------------------------------------------
# `ppforest2-core` aggregates OBJECT libraries that are safe to link on
# demand — their code gets pulled in naturally by whatever references it.
# Includes `ppforest2-models` (the Tree / Forest / TrainingSpec / etc.
# infrastructure); non-strategy model code doesn't have the self-
# registration hazard.
#
# Concrete strategies live in a separate STATIC `ppforest2-strategies`
# library (see `src/models/strategies/CMakeLists.txt`). That target is
# linked with WHOLE_ARCHIVE by every executable below so the linker
# can't dead-strip strategy object files whose only external effect is
# a static-init registration. Keeping strategies in their own archive
# means consumers that don't want every strategy's code pulled in (e.g.
# the R bindings skip the CSV reader entirely) still get surgical
# linkage for everything else in core.
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
  # `ppforest2-core` is wrapped in WHOLE_ARCHIVE so the linker keeps every
  # object file — even those whose only external effect is a static
  # initializer (the self-registering strategies via PPFOREST2_REGISTER_*).
  # Without this, ld64 / ld / lld dead-strip any strategy not directly
  # referenced from code linked into this executable, the registry stays
  # empty for that strategy, and `*::from_json` throws "Unknown strategy"
  # at runtime. CMake 3.24's generator expression handles the platform-
  # specific flag differences (macOS `-force_load`, GCC `--whole-archive`,
  # MSVC `/WHOLEARCHIVE`) uniformly.
  # `ppforest2-models` is WHOLE_ARCHIVE-linked so self-registering strategies
  # survive the linker's dead-strip pass — see the note on its STATIC split
  # above. `ppforest2-core` is linked normally; its objects pull in on
  # demand based on what this executable actually calls.
  # `httplib::httplib` is listed explicitly (rather than relying on
  # ppforest2-cli's PUBLIC link interface) because OBJECT libraries don't
  # propagate link deps through `$<TARGET_OBJECTS:>` extraction. Without it
  # the Windows MinGW build can't find the Winsock symbols cpp-httplib
  # brings in transitively (`ws2_32`).
  target_link_libraries(ppforest2 $<LINK_LIBRARY:WHOLE_ARCHIVE,ppforest2-strategies> ppforest2-core nlohmann_json::nlohmann_json csv CLI11::CLI11 fmt::fmt httplib::httplib)
  target_compile_definitions(ppforest2 PRIVATE PPFOREST2_VERSION="${PROJECT_VERSION}")

  # ---------------------------------------------------------------------------
  # Tests
  # ---------------------------------------------------------------------------
  enable_testing()

  set(PPFOREST2_TEST_SRC
    src/test.cpp
    src/models/Model.test.cpp
    src/models/Tree.test.cpp
    src/models/Forest.test.cpp
    src/models/TreeLeaf.test.cpp
    src/models/TreeBranch.test.cpp
    src/models/Bagged.test.cpp
    src/models/ClassificationTree.test.cpp
    src/models/ClassificationForest.test.cpp
    src/models/RegressionTree.test.cpp
    src/models/RegressionForest.test.cpp
    src/models/Evaluation.test.cpp
    src/models/strategies/pp/ProjectionPursuit.test.cpp
    src/models/strategies/pp/PDA.test.cpp
    src/models/strategies/vars/VariableSelection.test.cpp
    src/models/strategies/vars/Uniform.test.cpp
    src/models/strategies/vars/All.test.cpp
    src/models/strategies/cutpoint/Cutpoint.test.cpp
    src/models/strategies/cutpoint/MeanOfMeans.test.cpp
    src/models/strategies/stop/StopRule.test.cpp
    src/models/strategies/stop/PureNode.test.cpp
    src/models/strategies/stop/MinSize.test.cpp
    src/models/strategies/stop/MinVariance.test.cpp
    src/models/strategies/stop/CompositeStop.test.cpp
    src/models/strategies/stop/MaxDepth.test.cpp
    src/models/strategies/binarize/Binarization.test.cpp
    src/models/strategies/binarize/LargestGap.test.cpp
    src/models/strategies/binarize/Disabled.test.cpp
    src/models/strategies/grouping/Grouping.test.cpp
    src/models/strategies/grouping/ByLabel.test.cpp
    src/models/strategies/grouping/ByCutpoint.test.cpp
    src/models/strategies/leaf/MajorityVote.test.cpp
    src/models/strategies/leaf/MeanResponse.test.cpp
    src/models/TrainingSpec.test.cpp
    src/models/Visualization.test.cpp
    src/stats/Stats.test.cpp
    src/stats/GroupPartition.test.cpp
    src/stats/ConfusionMatrix.test.cpp
    src/stats/Metrics.test.cpp
    src/stats/Uniform.test.cpp
    src/stats/Normal.test.cpp
    src/stats/Simulation.test.cpp
    src/serialization/ExportValidation.test.cpp
    src/serialization/Json.test.cpp
    src/utils/Math.test.cpp
    src/utils/RangeVector.test.cpp
    src/utils/Invariant.test.cpp
    src/utils/JsonReader.test.cpp
    src/utils/System.test.cpp
    src/cli/CLIOptions.test.cpp
    src/cli/ModelParams.test.cpp
    src/cli/EvaluateParams.test.cpp
    src/cli/Train.integration.test.cpp
    src/cli/Predict.integration.test.cpp
    src/cli/Evaluate.integration.test.cpp
    src/cli/Summarize.integration.test.cpp
    src/cli/Benchmark.integration.test.cpp
    src/cli/Reproducibility.integration.test.cpp
    src/cli/CLI.integration.test.cpp
    src/cli/ServeHandlers.test.cpp
    src/cli/Serve.integration.test.cpp
    src/golden/Reproducibility.test.cpp
    src/golden/Threading.test.cpp
    src/cli/Validation.test.cpp
    src/cli/Benchmark.test.cpp
    src/cli/BenchmarkReport.test.cpp
    src/io/IO.test.cpp
    src/io/Presentation.test.cpp)

  add_executable(ppforest2-test ${PPFOREST2_TEST_SRC} $<TARGET_OBJECTS:ppforest2-cli>)

  target_link_libraries(ppforest2-test
    $<LINK_LIBRARY:WHOLE_ARCHIVE,ppforest2-strategies>  # see note on the ppforest2 target above
    ppforest2-core
    nlohmann_json::nlohmann_json
    gtest
    CLI11::CLI11
    csv
    fmt::fmt
    httplib::httplib)

  target_include_directories(ppforest2-test PRIVATE
    ${csv_SOURCE_DIR}/include
    ${gtest_SOURCE_DIR}/include
    ${gtest_SOURCE_DIR})

  target_compile_definitions(ppforest2-test PRIVATE
    PPFOREST2_BINARY_PATH="${CMAKE_BINARY_DIR}/ppforest2"
    PPFOREST2_DATA_DIR="${CMAKE_SOURCE_DIR}/../data"
    PPFOREST2_GOLDEN_DIR="${CMAKE_SOURCE_DIR}/../golden"
    PPFOREST2_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/../fixtures"
    PPFOREST2_BENCH_SCENARIOS_CLS="${CMAKE_SOURCE_DIR}/../bench/default-scenarios-classification.json"
    PPFOREST2_BENCH_SCENARIOS_REG="${CMAKE_SOURCE_DIR}/../bench/default-scenarios-regression.json"
    PPFOREST2_PLATFORM="${PPFOREST2_PLATFORM}"
    PPFOREST2_VERSION="${PROJECT_VERSION}")

  add_dependencies(ppforest2-test ppforest2)

  include(GoogleTest)
  gtest_discover_tests(ppforest2-test DISCOVERY_MODE PRE_TEST)

  # ---------------------------------------------------------------------------
  # Golden file generator
  # ---------------------------------------------------------------------------
  add_executable(ppforest2-golden-gen src/golden/GoldenGen.cpp)

  target_link_libraries(ppforest2-golden-gen
    $<LINK_LIBRARY:WHOLE_ARCHIVE,ppforest2-strategies>  # see note on the ppforest2 target above
    ppforest2-core
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
