set(SERVE_RESOURCES_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/serve_resources)
file(MAKE_DIRECTORY ${SERVE_RESOURCES_GEN_DIR})

set(SERVE_SUMMARY_HTML_HPP ${SERVE_RESOURCES_GEN_DIR}/summary_html.hpp)
set(SERVE_APP_CSS_HPP      ${SERVE_RESOURCES_GEN_DIR}/app_css.hpp)
set(SERVE_SUMMARY_JS_HPP   ${SERVE_RESOURCES_GEN_DIR}/summary_js.hpp)
set(SERVE_PREDICT_HTML_HPP ${SERVE_RESOURCES_GEN_DIR}/predict_html.hpp)

add_custom_command(
  OUTPUT ${SERVE_SUMMARY_HTML_HPP}
  COMMAND ${CMAKE_COMMAND}
    -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/resources/summary.html
    -DOUTPUT=${SERVE_SUMMARY_HTML_HPP}
    -DSYMBOL=SUMMARY_HTML
    -P ${CMAKE_SOURCE_DIR}/cmake/EmbedResource.cmake
  DEPENDS
    resources/summary.html
    ${CMAKE_SOURCE_DIR}/cmake/EmbedResource.cmake
  COMMENT "Embedding summary.html into ppforest2-cli"
  VERBATIM)

add_custom_command(
  OUTPUT ${SERVE_APP_CSS_HPP}
  COMMAND ${CMAKE_COMMAND}
    -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/resources/app.css
    -DOUTPUT=${SERVE_APP_CSS_HPP}
    -DSYMBOL=APP_CSS
    -P ${CMAKE_SOURCE_DIR}/cmake/EmbedResource.cmake
  DEPENDS
    resources/app.css
    ${CMAKE_SOURCE_DIR}/cmake/EmbedResource.cmake
  COMMENT "Embedding app.css into ppforest2-cli"
  VERBATIM)

add_custom_command(
  OUTPUT ${SERVE_SUMMARY_JS_HPP}
  COMMAND ${CMAKE_COMMAND}
    -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/resources/summary.js
    -DOUTPUT=${SERVE_SUMMARY_JS_HPP}
    -DSYMBOL=SUMMARY_JS
    -P ${CMAKE_SOURCE_DIR}/cmake/EmbedResource.cmake
  DEPENDS
    resources/summary.js
    ${CMAKE_SOURCE_DIR}/cmake/EmbedResource.cmake
  COMMENT "Embedding summary.js into ppforest2-cli"
  VERBATIM)

add_custom_command(
  OUTPUT ${SERVE_PREDICT_HTML_HPP}
  COMMAND ${CMAKE_COMMAND}
    -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/resources/predict.html
    -DOUTPUT=${SERVE_PREDICT_HTML_HPP}
    -DSYMBOL=PREDICT_HTML
    -P ${CMAKE_SOURCE_DIR}/cmake/EmbedResource.cmake
  DEPENDS
    resources/predict.html
    ${CMAKE_SOURCE_DIR}/cmake/EmbedResource.cmake
  COMMENT "Embedding predict.html into ppforest2-cli"
  VERBATIM)

add_library(ppforest2-cli OBJECT
  CLIOptions.cpp
  ModelParams.cpp
  EvaluateParams.cpp
  Validation.cpp
  Train.cpp
  Predict.cpp
  Evaluate.cpp
  Benchmark.cpp
  BenchmarkReport.cpp
  Summarize.cpp
  Serve.cpp
  ServeHandlers.cpp
  ${SERVE_SUMMARY_HTML_HPP}
  ${SERVE_APP_CSS_HPP}
  ${SERVE_SUMMARY_JS_HPP}
  ${SERVE_PREDICT_HTML_HPP})

target_include_directories(ppforest2-cli PRIVATE
  ${csv_SOURCE_DIR}/include
  ${SERVE_RESOURCES_GEN_DIR})

target_link_libraries(ppforest2-cli PUBLIC
  ppforest2-compile
  nlohmann_json::nlohmann_json
  CLI11::CLI11
  fmt::fmt
  httplib::httplib
  pantor::inja)

target_compile_definitions(ppforest2-cli PRIVATE
  PPFOREST2_VERSION="${PROJECT_VERSION}")
