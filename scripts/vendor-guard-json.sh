#!/bin/sh
# Bracket the vendored nlohmann json.hpp with a _Pragma guard that suppresses the
# libc++ char_traits<unsigned char> deprecation (Apple clang 21+, macOS 26 SDK)
# triggered by nlohmann's binary output/stream adapters (std::basic_string<uint8_t>,
# std::basic_ostream<uint8_t>). We never use nlohmann's binary formats, but the
# default template arguments are instantiated with nlohmann::json regardless.
# _Pragma (not #pragma) so R CMD check's pragma check does not flag it; clang-only
# so GCC does not warn on an unknown pragma. Run from r-vendor-deps after copying.
set -eu
json="$1"
tmp="$json.guard.tmp"
{
  cat <<'HDR'
// ppforest2: suppress libc++ char_traits<unsigned char> deprecation from
// nlohmann's binary adapters (see scripts/vendor-guard-json.sh).
#if defined(__clang__)
_Pragma("clang diagnostic push")
_Pragma("clang diagnostic ignored \"-Wdeprecated-declarations\"")
#endif
HDR
  cat "$json"
  cat <<'FTR'
#if defined(__clang__)
_Pragma("clang diagnostic pop")
#endif
FTR
} > "$tmp"
mv "$tmp" "$json"
