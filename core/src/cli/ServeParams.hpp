/**
 * @file ServeParams.hpp
 * @brief Serve subcommand parameters.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace ppforest2::cli {
  /** @brief `serve` subcommand options. CLI-exclusive; no config file roundtrip. */
  struct ServeParams {
    std::string host           = "127.0.0.1";
    uint16_t port              = 8080;
    std::size_t max_body_bytes = 1024UL * 1024UL;
  };
}
