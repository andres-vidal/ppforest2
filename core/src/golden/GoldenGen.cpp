/**
 * @file GoldenGen.cpp
 * @brief Generates golden reference files for reproducibility testing.
 *
 * For each configuration (dataset × model params), trains a model with a
 * fixed seed, computes predictions, metrics, and variable importance, then
 * writes the results as a JSON golden file.
 */
#include "utils/Types.hpp"
#include "stats/DataPacket.hpp"
#include "stats/ConfusionMatrix.hpp"
#include "models/Model.hpp"
#include "models/TrainingSpec.hpp"
#include "serialization/Json.hpp"
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
static std::string const GOLDEN_DIR = PPFOREST2_GOLDEN_DIR;
static std::string const PLATFORM   = PPFOREST2_PLATFORM;

struct GoldenConfig {
  std::string csv_path;
  int size; // 0 = single tree
  Feature lambda;
  int n_vars; // ignored for single tree
  int seed;

  std::string dataset() const { return std::filesystem::path(csv_path).stem().string(); }

  std::string slug() const {
    std::string s;

    if (size > 0) {
      s = "forest-pda";

      if (lambda > 0) {
        std::string l = fmt::format("{:g}", lambda);
        l.erase(std::remove(l.begin(), l.end(), '.'), l.end());
        s += fmt::format("-l{}", l);
      }

      s += fmt::format("-n{}", size);
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
namespace {
  void generate_golden(GoldenConfig const& config) {
    std::string const dir  = GOLDEN_DIR + "/" + config.dataset();
    std::string const path = dir + "/" + config.slug() + ".json";

    std::filesystem::create_directories(dir);

    DataPacket data = io::csv::read_sorted(config.csv_path);

    // Train
    auto vars = (config.size > 0) ? vars::uniform(config.n_vars) : vars::all();

    TrainingSpec const spec = TrainingSpec::builder(Mode::Classification)
                                  .size(config.size)
                                  .seed(config.seed)
                                  .threads(1)
                                  .pp(pp::pda(config.lambda))
                                  .vars(std::move(vars))
                                  .build();

    auto model = Model::train(spec, data.x, data.y);

    // Serialize model + meta + config + metrics
    serialization::Export<Model::Ptr> model_export{
        std::move(model),
        data.group_names,
        nullptr,
        static_cast<int>(data.x.rows()),
        static_cast<int>(data.x.cols()),
        data.feature_names,
    };

    model_export.compute_metrics(data.x, data.y);

    json result                  = model_export.to_json();
    result["config"]["platform"] = PLATFORM;
    result["config"]["dataset"]  = config.dataset();

    OutcomeVector const predictions     = model_export.model->predict(data.x);
    result["predictions"]               = serialization::to_json(predictions, data.group_names);
    GroupIdVector const predictions_int = predictions.cast<GroupId>();
    result["vote_proportions"]          = to_json(predict_proportions(*model_export.model, data.x));

    GroupIdVector const y_int = data.y.cast<GroupId>();
    ConfusionMatrix const cm(predictions_int, y_int);
    result["error_rate"] = cm.error();

    io::json::write_file(result, path);
    fmt::print("  {} -> {}\n", config.slug(), path);
  }
}

int main(int argc, char* argv[]) try {
  std::string output_dir = (argc > 1) ? argv[1] : GOLDEN_DIR;

  fmt::print("Generating golden files in: {}\n", output_dir);
  fmt::print("Platform: {}\n\n", PLATFORM);

  std::vector<GoldenConfig> const configs = {
      {DATA_DIR + "/classification/iris.csv", 0, 0.0F, 0, 0},
      {DATA_DIR + "/classification/iris.csv", 5, 0.0F, 2, 0},
      {DATA_DIR + "/classification/iris.csv", 5, 0.5F, 2, 0},
      {DATA_DIR + "/classification/crab.csv", 0, 0.0F, 0, 0},
      {DATA_DIR + "/classification/crab.csv", 10, 0.0F, 3, 0},
      {DATA_DIR + "/classification/wine.csv", 10, 0.0F, 4, 0},
      {DATA_DIR + "/classification/glass.csv", 10, 0.0F, 3, 0},
  };

  for (auto const& config : configs) {
    generate_golden(config);
  }

  fmt::print("\n{}\n", success("Done. " + emphasis(std::to_string(configs.size())) + " golden files generated."));
  return 0;
} catch (std::exception const& e) {
  fmt::print(stderr, "Error: {}\n", e.what());
  return 1;
}
