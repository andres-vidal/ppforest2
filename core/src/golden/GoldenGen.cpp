/**
 * @file GoldenGen.cpp
 * @brief Generates golden reference files for reproducibility testing.
 *
 * For each configuration (dataset × model params), trains a model with a
 * fixed seed, computes predictions, metrics, and variable importance, then
 * writes the results as a JSON golden file.
 */
#include "ppforest2.hpp"

#include "utils/Types.hpp"
#include "stats/DataPacket.hpp"
#include "stats/Stats.hpp"
#include "stats/ConfusionMatrix.hpp"
#include "models/Tree.hpp"
#include "models/Forest.hpp"
#include "models/TrainingSpecPDA.hpp"
#include "models/TrainingSpecUPDA.hpp"
#include "models/VariableImportance.hpp"
#include "serialization/Json.hpp"
#include "cli/Metrics.hpp"
#include "io/Color.hpp"
#include "io/IO.hpp"

#include <nlohmann/json.hpp>
#include <fmt/format.h>

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

using namespace ppforest2;
using namespace ppforest2::types;
using namespace ppforest2::stats;
using namespace ppforest2::io::style;
using namespace ppforest2::serialization;
using json = nlohmann::json;

#ifndef PPFOREST2_DATA_DIR
#error "PPFOREST2_DATA_DIR must be defined"
#endif

#ifndef PPFOREST2_GOLDEN_DIR
#error "PPFOREST2_GOLDEN_DIR must be defined"
#endif

#ifndef PPFOREST2_PLATFORM
#error "PPFOREST2_PLATFORM must be defined"
#endif

static const std::string DATA_DIR   = PPFOREST2_DATA_DIR;
static const std::string GOLDEN_DIR = PPFOREST2_GOLDEN_DIR;
static const std::string PLATFORM   = PPFOREST2_PLATFORM;

struct GoldenConfig {
  std::string csv_path;
  int trees;                // 0 = single tree
  float lambda;
  int n_vars;               // ignored for single tree
  int seed;

  std::string dataset() const {
    return std::filesystem::path(csv_path).stem().string();
  }

  std::string slug() const {
    std::string s;

    if (trees > 0) {
      s = (lambda > 0) ? "forest-pda" : "forest-pda";

      if (lambda > 0) {
        std::string l = fmt::format("{:g}", lambda);
        l.erase(std::remove(l.begin(), l.end(), '.'), l.end());
        s += fmt::format("-l{}", l);
      }

      s += fmt::format("-t{}", trees);
    } else {
      s = "tree-pda";
    }

    s += fmt::format("-s{}", seed);
    return s;
  }
};

/**
 * @brief Generate a golden file for a given configuration.
 *
 * Trains a model with the given configuration, computes predictions, metrics,
 * and variable importance, then writes the results as a JSON golden file .
 *
 * @param config The configuration to generate a golden file for.
 */
static void generate_golden(const GoldenConfig& config) {
  std::string dir  = GOLDEN_DIR + "/" + config.dataset();
  std::string path = dir + "/" + config.slug() + ".json";

  std::filesystem::create_directories(dir);

  DataPacket data = io::csv::read_sorted(config.csv_path);

  // Build config JSON — shared fields plus golden-specific (platform, dataset).
  json cfg;
  cfg["trees"]    = config.trees;
  cfg["lambda"]   = config.lambda;
  cfg["seed"]     = config.seed;
  cfg["platform"] = PLATFORM;
  cfg["dataset"]  = config.dataset();

  if (config.trees > 0) {
    cfg["vars"] = config.n_vars;
  }

  // Train
  Model::Ptr model;

  if (config.trees > 0) {
    model = Forest::make(
      TrainingSpecUPDA(config.n_vars, config.lambda),
      data.x, data.y, config.trees, config.seed, 1);
  } else {
    RNG rng(config.seed);
    model = Tree::make(
      TrainingSpecPDA(config.lambda), data.x, data.y, rng);
  }

  // Serialize model + meta + config
  json result = serialization::build_model_json(
    *model, cfg, data.group_names, data.feature_names,
    static_cast<int>(data.x.rows()), static_cast<int>(data.x.cols()));

  // Metrics (VI, confusion matrices, OOB)
  cli::compute_metrics(result, *model, data.x, data.y, data.group_names, config.seed);

  // Golden-specific fields
  ResponseVector predictions = model->predict(data.x);
  result["predictions"]      = serialization::to_labels(predictions, data.group_names);
  result["vote_proportions"] = to_json(model->predict(data.x, Proportions{}));

  ConfusionMatrix cm(predictions, data.y);
  result["error_rate"] = cm.error();

  io::json::write_file(result, path);
  fmt::print("  {} -> {}\n", config.slug(), path);
}

int main(int argc, char *argv[]) {
  std::string output_dir = (argc > 1) ? argv[1] : GOLDEN_DIR;

  fmt::print("Generating golden files in: {}\n", output_dir);
  fmt::print("Platform: {}\n\n", PLATFORM);

  std::vector<GoldenConfig> configs = {
    { DATA_DIR + "/iris.csv",  0,  0.0f, 0, 42 },
    { DATA_DIR + "/iris.csv",  5,  0.0f, 2, 42 },
    { DATA_DIR + "/iris.csv",  5,  0.5f, 2, 42 },
    { DATA_DIR + "/crab.csv",  0,  0.0f, 0, 42 },
    { DATA_DIR + "/crab.csv",  10, 0.0f, 3, 42 },
    { DATA_DIR + "/wine.csv",  10, 0.0f, 4, 42 },
    { DATA_DIR + "/glass.csv", 10, 0.0f, 3, 42 },
  };

  for (const auto& config : configs) {
    generate_golden(config);
  }

  fmt::print("\n{}\n", success("Done. " + emphasis(std::to_string(configs.size())) + " golden files generated."));
  return 0;
}
