/**
 * @file Serve.integration.test.cpp
 * @brief Smoke tests for the `serve` subcommand wiring.
 *
 * These tests verify CLI parsing and error paths without starting the HTTP
 * server (binding to ports may not be permitted in every environment).
 * Network-level coverage of the handlers themselves is in
 * `ServeHandlers.test.cpp`.
 */
#include "cli/CLI.integration.hpp"

#include <gtest/gtest.h>

TEST(ServeCLI, HelpListsServeSubcommand) {
  auto r = run_ppforest2("--help");
  EXPECT_EQ(r.exit_code, 0);
  EXPECT_NE(r.stdout_output.find("serve"), std::string::npos);
}

TEST(ServeCLI, ServeHelpDescribesEndpoints) {
  auto r = run_ppforest2("serve --help");
  EXPECT_EQ(r.exit_code, 0);
  EXPECT_NE(r.stdout_output.find("Serve predictions over HTTP"), std::string::npos);
  EXPECT_NE(r.stdout_output.find("--model"), std::string::npos);
  EXPECT_NE(r.stdout_output.find("--host"), std::string::npos);
  EXPECT_NE(r.stdout_output.find("--port"), std::string::npos);
}

TEST(ServeCLI, FailsWithoutModel) {
  auto r = run_ppforest2("serve --port 8080");
  EXPECT_NE(r.exit_code, 0);
  EXPECT_NE(r.stderr_output.find("--model"), std::string::npos);
}

TEST(ServeCLI, RejectsMissingModelFile) {
  auto r = run_ppforest2("serve --model /nonexistent/path.json");
  EXPECT_NE(r.exit_code, 0);
}

TEST(ServeCLI, RejectsOutOfRangePort) {
  auto r = run_ppforest2("serve --model /tmp/x.json --port 99999");
  EXPECT_NE(r.exit_code, 0);
}
