# EmbedResource.cmake
#
# Generates a C++ header that exposes a file's contents as a raw string
# literal. Invoked at build time via `cmake -P`.
#
# Required cache variables (passed via `-D`):
#   INPUT  — path to the resource file to embed
#   OUTPUT — path to the .hpp file to write
#   SYMBOL — name of the `char const[]` constant in the generated header
#
# The constant lives in `ppforest2::cli::serve::resources`.

if(NOT INPUT OR NOT OUTPUT OR NOT SYMBOL)
  message(FATAL_ERROR "EmbedResource.cmake requires -DINPUT=, -DOUTPUT=, -DSYMBOL=")
endif()

file(READ "${INPUT}" CONTENT)

# Long, unlikely delimiter so the raw string never closes early.
set(DELIM "PPFOREST2_EMBED")

if(CONTENT MATCHES "\\)${DELIM}\"")
  message(FATAL_ERROR
    "Resource file ${INPUT} contains the raw-string delimiter `)${DELIM}\"`; "
    "pick a different DELIM in EmbedResource.cmake")
endif()

get_filename_component(INPUT_NAME "${INPUT}" NAME)

file(WRITE "${OUTPUT}"
"// Generated from ${INPUT_NAME}. Do not edit.
#pragma once

namespace ppforest2::cli::serve::resources {
  inline constexpr char const ${SYMBOL}[] = R\"${DELIM}(${CONTENT})${DELIM}\";
}
")
