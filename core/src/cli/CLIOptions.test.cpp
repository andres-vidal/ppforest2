/**
 * @file CLIOptions.test.cpp
 * @brief Unit tests for CLI argument parsing (parse_args), parameter
 *        initialization (init_params), unused-parameter warnings
 *        (warn_unused_params), ModelStats aggregation, and file helpers.
 *
 * Tests exercise the in-process parsing path by constructing argv arrays
 * and calling parse_args() directly.  EXPECT_THROW tests verify
 * that invalid inputs throw UserError.
 */
#include <gtest/gtest.h>

#include "cli/CLIOptions.hpp"
#include "utils/UserError.hpp"
#include "io/Output.hpp"


using namespace ppforest2;
using namespace ppforest2::cli;
using namespace ppforest2::io;


#ifndef PPFOREST2_DATA_DIR
#error "PPFOREST2_DATA_DIR must be defined"
#endif

static const std::string IRIS_PATH = std::string(PPFOREST2_DATA_DIR) + "/classification/iris.csv";

namespace {
  /**
   * @brief Build an argv vector and call parse_args().
   *
   * Convenience wrapper so tests can write:
   *   auto opts = parse({ "ppforest2", "train", "-d", "data.csv" });
   *
   * @param args_list Initializer list of C-string arguments (first must be program name).
   * @return Populated Params struct.
   */
  Params parse(std::initializer_list<char const*> args_list) {
    std::vector<char const*> args(args_list);
    return parse_args(static_cast<int>(args.size()), const_cast<char**>(args.data()));
  }

}

// ---------------------------------------------------------------------------
// parse_args() — subcommand routing
// ---------------------------------------------------------------------------

/* Parsing "train" selects the train subcommand. */
TEST(ParseArgs, TrainSubcommand) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str()});
  EXPECT_EQ(opts.subcommand, Subcommand::train);
}

/* Parsing "evaluate" with --simulate selects the evaluate subcommand. */
TEST(ParseArgs, EvaluateSubcommandWithSimulate) {
  auto opts = parse({"ppforest2", "evaluate", "--simulate", "100x5x2"});
  EXPECT_EQ(opts.subcommand, Subcommand::evaluate);
}

/* Parsing "evaluate" with -d selects evaluate and captures the data path. */
TEST(ParseArgs, EvaluateSubcommandWithData) {
  auto opts = parse({"ppforest2", "evaluate", "-d", IRIS_PATH.c_str()});
  EXPECT_EQ(opts.subcommand, Subcommand::evaluate);
  EXPECT_EQ(opts.data_path, IRIS_PATH);
}

/* Omitting a subcommand must throw. */
TEST(ParseArgs, NoSubcommandExits) {
  EXPECT_THROW(parse({"ppforest2"}), ppforest2::UserError);
}

/* An unrecognised subcommand must throw. */
TEST(ParseArgs, InvalidSubcommandExits) {
  EXPECT_THROW(parse({"ppforest2", "foobar"}), ppforest2::UserError);
}

// ---------------------------------------------------------------------------
// parse_args() — train defaults and options
// ---------------------------------------------------------------------------

/* Verify all default values when only -d is supplied. */
TEST(ParseArgs, TrainDefaultValues) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str()});
  EXPECT_EQ(opts.model.size, 100);
  EXPECT_FLOAT_EQ(opts.model.lambda, 0.5f);
  EXPECT_FALSE(opts.model.threads.has_value());
  EXPECT_FALSE(opts.model.seed.has_value());
  EXPECT_FALSE(opts.model.p_vars.has_value());
  EXPECT_FALSE(opts.model.n_vars.has_value());
  EXPECT_TRUE(opts.model.vars_input.empty());
  EXPECT_EQ(opts.save_path, "model.json");
  EXPECT_FALSE(opts.no_save);
}

/* -n overrides the default tree count. */
TEST(ParseArgs, TrainSizeOption) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "-n", "50"});
  EXPECT_EQ(opts.model.size, 50);
}

/* -l overrides the default lambda. */
TEST(ParseArgs, TrainLambdaOption) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "-l", "0.3"});
  EXPECT_FLOAT_EQ(opts.model.lambda, 0.3F);
}

/* --threads sets the thread count explicitly. */
TEST(ParseArgs, TrainThreadsOption) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--threads", "4"});
  EXPECT_EQ(opts.model.threads.value(), 4);
}

/* -r sets the random seed. */
TEST(ParseArgs, TrainSeedOption) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "-r", "0"});
  EXPECT_EQ(opts.model.seed.value(), 0);
}

// ---------------------------------------------------------------------------
// parse_args() — --n-vars and --p-vars (feature sub-sampling) parsing
// ---------------------------------------------------------------------------

/* --p-vars with a decimal value (0.8) is interpreted as a proportion. */
TEST(ParseArgs, PVarsAsProportion) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--p-vars", "0.8"});
  EXPECT_FLOAT_EQ(opts.model.p_vars.value(), 0.8F);
  EXPECT_FALSE(opts.model.n_vars.has_value());
}

/* --n-vars with an integer (3) is interpreted as an absolute feature count. */
TEST(ParseArgs, NVarsAsAbsoluteCount) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--n-vars", "3"});
  EXPECT_EQ(opts.model.n_vars.value(), 3);
  EXPECT_FALSE(opts.model.p_vars.has_value());
}

/* --p-vars with a fraction "1/3" is parsed as a proportion. */
TEST(ParseArgs, PVarsAsFraction) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--p-vars", "1/3"});
  EXPECT_NEAR(opts.model.p_vars.value(), 1.0F / 3.0F, 0.001F);
  EXPECT_FALSE(opts.model.n_vars.has_value());
}

/* --p-vars "1/2" parses to exactly 0.5. */
TEST(ParseArgs, PVarsAsFractionHalf) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--p-vars", "1/2"});
  EXPECT_FLOAT_EQ(opts.model.p_vars.value(), 0.5F);
  EXPECT_FALSE(opts.model.n_vars.has_value());
}

/* --p-vars "3/3" parses to 1.0 (use all features). */
TEST(ParseArgs, PVarsAsFractionFull) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--p-vars", "3/3"});
  EXPECT_FLOAT_EQ(opts.model.p_vars.value(), 1.0F);
  EXPECT_FALSE(opts.model.n_vars.has_value());
}

/* --p-vars with fraction > 1 (4/3) must throw. */
TEST(ParseArgs, PVarsFractionGreaterThanOneExits) {
  EXPECT_THROW(parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--p-vars", "4/3"}), ppforest2::UserError);
}

/* --p-vars with division by zero (1/0) must throw. */
TEST(ParseArgs, PVarsFractionZeroDenominatorExits) {
  EXPECT_THROW(parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--p-vars", "1/0"}), ppforest2::UserError);
}

/* --p-vars with zero numerator (0/3) must throw — no features selected. */
TEST(ParseArgs, PVarsFractionZeroNumeratorExits) {
  EXPECT_THROW(parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--p-vars", "0/3"}), ppforest2::UserError);
}

/* --p-vars with non-numeric fraction ("a/b") must throw. */
TEST(ParseArgs, PVarsFractionInvalidExits) {
  EXPECT_THROW(parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--p-vars", "a/b"}), ppforest2::UserError);
}

/* --n-vars and --p-vars are mutually exclusive. */
TEST(ParseArgs, NVarsExcludesPVars) {
  EXPECT_THROW(
      parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--n-vars", "3", "--p-vars", "0.5"}), ppforest2::UserError
  );
}

// ---------------------------------------------------------------------------
// parse_args() — save, no-save, and lambda validation
// ---------------------------------------------------------------------------

/* -s sets a custom model output path. */
TEST(ParseArgs, TrainSaveOption) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "-s", "/tmp/model.json"});
  EXPECT_EQ(opts.save_path, "/tmp/model.json");
}

/* Zero trees is accepted (means a single projection-pursuit tree). */
TEST(ParseArgs, TrainZeroTrees) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "-n", "0"});
  EXPECT_EQ(opts.model.size, 0);
}

/* Train with a nonexistent data file must throw. */
TEST(ParseArgs, TrainNonexistentDataExits) {
  EXPECT_THROW(parse({"ppforest2", "train", "-d", "/nonexistent/file.csv"}), ppforest2::UserError);
}

/* Lambda = 0 is a valid boundary value (pure LDA). */
TEST(ParseArgs, TrainLambdaZeroValid) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "-l", "0"});
  EXPECT_FLOAT_EQ(opts.model.lambda, 0.0F);
}

/* Lambda = 1 is a valid boundary value (pure PCA). */
TEST(ParseArgs, TrainLambdaOneValid) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "-l", "1.0"});
  EXPECT_FLOAT_EQ(opts.model.lambda, 1.0F);
}

/* --no-save disables model saving and clears the save path. */
TEST(ParseArgs, TrainNoSaveFlag) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--no-save"});
  EXPECT_TRUE(opts.no_save);
}

/* --no-save and -s together must throw (mutually exclusive). */
TEST(ParseArgs, TrainNoSaveExcludesSave) {
  EXPECT_THROW(
      parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--no-save", "-s", "/tmp/m.json"}), ppforest2::UserError
  );
}

// ---------------------------------------------------------------------------
// parse_args() — predict options
// ---------------------------------------------------------------------------

/* -o sets the prediction output file path. */
TEST(ParseArgs, PredictOutputOption) {
  auto opts =
      parse({"ppforest2", "predict", "-M", IRIS_PATH.c_str(), "-d", IRIS_PATH.c_str(), "-o", "/tmp/results.json"});
  EXPECT_EQ(opts.output_path, "/tmp/results.json");
}

/* --no-metrics flag is captured. */
TEST(ParseArgs, PredictNoMetricsFlag) {
  auto opts = parse({"ppforest2", "predict", "-M", IRIS_PATH.c_str(), "-d", IRIS_PATH.c_str(), "--no-metrics"});
  EXPECT_TRUE(opts.no_metrics);
}

/* --no-proportions flag is captured. */
TEST(ParseArgs, PredictNoProportionsFlag) {
  auto opts = parse({"ppforest2", "predict", "-M", IRIS_PATH.c_str(), "-d", IRIS_PATH.c_str(), "--no-proportions"});
  EXPECT_TRUE(opts.no_proportions);
}

/* --no-proportions defaults to false (proportions included by default). */
TEST(ParseArgs, PredictNoProportionsDefault) {
  auto opts = parse({"ppforest2", "predict", "-M", IRIS_PATH.c_str(), "-d", IRIS_PATH.c_str()});
  EXPECT_FALSE(opts.no_proportions);
}

/* Predict without -M (model path) must throw. */
TEST(ParseArgs, PredictMissingModelExits) {
  EXPECT_THROW(parse({"ppforest2", "predict", "-d", IRIS_PATH.c_str()}), ppforest2::UserError);
}

/* Predict without -d (data path) must throw. */
TEST(ParseArgs, PredictMissingDataExits) {
  EXPECT_THROW(parse({"ppforest2", "predict", "-M", IRIS_PATH.c_str()}), ppforest2::UserError);
}

// ---------------------------------------------------------------------------
// parse_args() — evaluate options
// ---------------------------------------------------------------------------

/* -o sets the evaluation output file path. */
TEST(ParseArgs, EvaluateOutputOption) {
  auto opts = parse({"ppforest2", "evaluate", "--simulate", "100x5x2", "-o", "/tmp/results.json"});
  EXPECT_EQ(opts.output_path, "/tmp/results.json");
}

/* Evaluate must reject -s (--save is train-only). */
TEST(ParseArgs, EvaluateNoSaveOption) {
  // Evaluate should not accept -s
  EXPECT_THROW(
      parse({"ppforest2", "evaluate", "--simulate", "100x5x2", "-s", "/tmp/model.json"}), ppforest2::UserError
  );
}

/* Evaluate always has an empty save_path (save is not available). */
/* Evaluate never uses save_path — it stays at its default. */
TEST(ParseArgs, EvaluateDoesNotUseSavePath) {
  auto opts = parse({"ppforest2", "evaluate", "--simulate", "100x5x2"});
  EXPECT_FALSE(opts.no_save);
}

/* -e sets the experiment export directory. */
TEST(ParseArgs, EvaluateExportOption) {
  auto opts = parse({"ppforest2", "evaluate", "--simulate", "100x5x2", "-e", "/tmp/experiment"});
  EXPECT_EQ(opts.evaluate.export_path, "/tmp/experiment");
}

/* --simulate "RxCxK" is parsed into rows, cols, and groups. */
TEST(ParseArgs, EvaluateSimulateFormat) {
  auto opts = parse({"ppforest2", "evaluate", "--simulate", "100x5x2"});
  EXPECT_EQ(opts.simulation.rows, 100);
  EXPECT_EQ(opts.simulation.cols, 5);
  EXPECT_EQ(opts.simulation.n_groups, 2);
}

/* Simulation generation parameters (mean, separation, sd) are captured. */
TEST(ParseArgs, EvaluateSimulateCustomParams) {
  auto opts = parse(
      {"ppforest2",
       "evaluate",
       "--simulate",
       "100x5x3",
       "--simulate-mean",
       "200",
       "--simulate-mean-separation",
       "25",
       "--simulate-sd",
       "5"}
  );
  EXPECT_FLOAT_EQ(opts.simulation.classification.mean, 200.0F);
  EXPECT_FLOAT_EQ(opts.simulation.classification.mean_separation, 25.0F);
  EXPECT_FLOAT_EQ(opts.simulation.classification.sd, 5.0F);
  EXPECT_FLOAT_EQ(opts.simulation.regression.sd, 5.0F);
}

/* -p sets the train/test split ratio. */
TEST(ParseArgs, EvaluateTrainRatio) {
  auto opts = parse({"ppforest2", "evaluate", "--simulate", "100x5x2", "-p", "0.8"});
  EXPECT_FLOAT_EQ(opts.evaluate.train_ratio.value(), 0.8F);
}

/* -i sets fixed iteration count and disables convergence. */
TEST(ParseArgs, EvaluateIterations) {
  auto opts = parse({"ppforest2", "evaluate", "--simulate", "100x5x2", "-i", "5"});
  EXPECT_EQ(opts.evaluate.iterations.value(), 5);
  EXPECT_FALSE(opts.evaluate.convergence_enabled());
}


/* Providing both -d and --simulate must exit (mutually exclusive). */
TEST(ParseArgs, EvaluateBothDataAndSimulateExits) {
  EXPECT_THROW(
      parse({"ppforest2", "evaluate", "-d", IRIS_PATH.c_str(), "--simulate", "100x5x2"}), ppforest2::UserError
  );
}

// ---------------------------------------------------------------------------
// parse_args() — convergence defaults and -i override
// ---------------------------------------------------------------------------

/* Evaluate defaults to convergence mode. */
TEST(ParseArgs, EvaluateDefaultConvergence) {
  auto opts = parse({"ppforest2", "evaluate", "--simulate", "100x5x2"});
  opts.evaluate.resolve_defaults();
  EXPECT_TRUE(opts.evaluate.convergence_enabled());
  EXPECT_EQ(opts.evaluate.convergence.max.value(), 200);
  EXPECT_FLOAT_EQ(*opts.evaluate.convergence.cv, 0.05F);
  EXPECT_EQ(opts.evaluate.convergence.min.value(), 10);
  EXPECT_EQ(opts.evaluate.convergence.window.value(), 3);
}

/* -i disables convergence and sets fixed iteration count. */
TEST(ParseArgs, IterationsDisablesConvergence) {
  auto opts = parse({"ppforest2", "evaluate", "--simulate", "100x5x2", "-i", "5"});
  EXPECT_FALSE(opts.evaluate.convergence_enabled());
  EXPECT_EQ(opts.evaluate.iterations.value(), 5);
}

/* --convergence-max overrides convergence cap without disabling it. */
TEST(ParseArgs, MaxIterationsOverride) {
  auto opts = parse({"ppforest2", "evaluate", "--simulate", "100x5x2", "--convergence-max", "500"});
  EXPECT_TRUE(opts.evaluate.convergence_enabled());
  EXPECT_EQ(opts.evaluate.convergence.max.value(), 500);
}

/* --convergence-cv overrides the convergence threshold. */
TEST(ParseArgs, CvThresholdOverride) {
  auto opts = parse({"ppforest2", "evaluate", "--simulate", "100x5x2", "--convergence-cv", "0.01"});
  EXPECT_TRUE(opts.evaluate.convergence_enabled());
  EXPECT_FLOAT_EQ(opts.evaluate.convergence.cv.value(), 0.01F);
}

/* --convergence-min overrides the minimum before convergence checking. */
TEST(ParseArgs, MinIterationsOverride) {
  auto opts = parse({"ppforest2", "evaluate", "--simulate", "100x5x2", "--convergence-min", "20"});
  EXPECT_TRUE(opts.evaluate.convergence_enabled());
  EXPECT_EQ(opts.evaluate.convergence.min.value(), 20);
}

/* --convergence-window overrides the consecutive stable iterations required. */
TEST(ParseArgs, StableWindowOverride) {
  auto opts = parse({"ppforest2", "evaluate", "--simulate", "100x5x2", "--convergence-window", "5"});
  EXPECT_TRUE(opts.evaluate.convergence_enabled());
  EXPECT_EQ(opts.evaluate.convergence.window.value(), 5);
}

/* All convergence parameters can be set together. */
TEST(ParseArgs, AllConvergenceParams) {
  auto opts = parse(
      {"ppforest2",
       "evaluate",
       "--simulate",
       "100x5x2",
       "--convergence-cv",
       "0.02",
       "--convergence-max",
       "300",
       "--convergence-min",
       "15",
       "--convergence-window",
       "4"}
  );
  EXPECT_TRUE(opts.evaluate.convergence_enabled());
  EXPECT_FLOAT_EQ(opts.evaluate.convergence.cv.value(), 0.02F);
  EXPECT_EQ(opts.evaluate.convergence.max.value(), 300);
  EXPECT_EQ(opts.evaluate.convergence.min.value(), 15);
  EXPECT_EQ(opts.evaluate.convergence.window.value(), 4);
}

/* -i and convergence options are mutually exclusive. */
TEST(ParseArgs, IterationsExcludesConvergenceParams) {
  EXPECT_THROW(
      parse({"ppforest2", "evaluate", "--simulate", "100x5x2", "-i", "10", "--convergence-cv", "0.01"}),
      ppforest2::UserError
  );
}

/* Simulation parameters without --simulate must exit. */
TEST(ParseArgs, EvaluateSimParamsNeedSimulate) {
  EXPECT_THROW(
      parse({"ppforest2", "evaluate", "-d", IRIS_PATH.c_str(), "--simulate-mean", "200"}), ppforest2::UserError
  );
}

// ---------------------------------------------------------------------------
// parse_args() — global flags
// ---------------------------------------------------------------------------

/* --no-color flag is captured when placed before the subcommand. */
TEST(ParseArgs, NoColorFlag) {
  auto opts = parse({"ppforest2", "--no-color", "evaluate", "--simulate", "100x5x2"});
  EXPECT_TRUE(opts.no_color);
}

/* --no-color defaults to false. */
TEST(ParseArgs, NoColorDefaultFalse) {
  auto opts = parse({"ppforest2", "evaluate", "--simulate", "100x5x2"});
  EXPECT_FALSE(opts.no_color);
}

/* -q enables quiet mode. */
TEST(ParseArgs, QuietShortFlag) {
  auto opts = parse({"ppforest2", "-q", "evaluate", "--simulate", "100x5x2"});
  EXPECT_TRUE(opts.quiet);
}

/* --quiet also enables quiet mode. */
TEST(ParseArgs, QuietLongFlag) {
  auto opts = parse({"ppforest2", "--quiet", "evaluate", "--simulate", "100x5x2"});
  EXPECT_TRUE(opts.quiet);
}


/* --version prints version info and exits with code 0. */
TEST(ParseArgs, VersionExits) {
  EXPECT_EXIT(parse({"ppforest2", "--version"}), testing::ExitedWithCode(0), "");
}

/* --help prints usage info and exits with code 0. */
TEST(ParseArgs, HelpExits) {
  EXPECT_EXIT(parse({"ppforest2", "--help"}), testing::ExitedWithCode(0), "");
}

// ---------------------------------------------------------------------------
// resolve_defaults() — seed and threads resolution
// ---------------------------------------------------------------------------

/* Unset seed is auto-generated. */
TEST(ResolveDefaults, UnsetSeedGetsGenerated) {
  Params params;
  EXPECT_FALSE(params.model.seed.has_value());
  params.resolve_seed();
  EXPECT_TRUE(params.model.seed.has_value());
}

/* An explicit seed is preserved. */
TEST(ResolveDefaults, ExplicitSeedPreserved) {
  Params params;
  params.model.seed = 0;
  params.resolve_seed();
  EXPECT_EQ(params.model.seed.value(), 0);
}

/* Calling resolve_seed twice preserves the first generated seed. */
TEST(ResolveDefaults, ResolveSeedIsIdempotent) {
  Params params;
  params.resolve_seed();
  auto seed = params.model.seed.value();
  params.resolve_seed();
  EXPECT_EQ(params.model.seed.value(), seed);
}

/* resolve_defaults does not override a previously resolved seed. */
TEST(ResolveDefaults, ResolveDefaultsPreservesResolvedSeed) {
  Params params;
  params.resolve_seed();
  auto seed = params.model.seed.value();
  params.resolve_defaults(0);
  EXPECT_EQ(params.model.seed.value(), seed);
}

/* Unset threads is auto-detected. */
TEST(ResolveDefaults, UnsetThreadsGetsDetected) {
  Params params;
  EXPECT_FALSE(params.model.threads.has_value());
  params.resolve_defaults(0);
  EXPECT_TRUE(params.model.threads.has_value());
  EXPECT_GE(params.model.threads.value(), 1);
}

/* An explicit thread count is preserved. */
TEST(ResolveDefaults, ExplicitThreadsPreserved) {
  Params params;
  params.model.threads = 8;
  params.resolve_defaults(0);
  EXPECT_EQ(params.model.threads.value(), 8);
}

// ---------------------------------------------------------------------------
// resolve_defaults() — default resolution
// ---------------------------------------------------------------------------

/* Default lambda (0.5) is preserved when not overridden. */
TEST(ResolveDefaults, DefaultLambda) {
  Params params;
  EXPECT_FLOAT_EQ(params.model.lambda, 0.5F);
}

/* An explicitly set lambda is preserved. */
TEST(ResolveDefaults, LambdaUnchangedIfSet) {
  Params params;
  params.model.lambda = 0.3F;
  params.resolve_defaults(0);
  EXPECT_FLOAT_EQ(params.model.lambda, 0.3F);
}

/* Unset vars get defaults (p_vars=0.5, n_vars computed). */
TEST(ResolveDefaults, UnsetVarsGetDefaults) {
  // Mode must be set before `resolve_defaults` — in production
  // `Params::read_data` deduces it from the training data; tests
  // bypassing that path must set it explicitly.
  Params params;
  params.model.mode_input = "classification";
  params.model.size       = 10;
  params.quiet            = true;
  EXPECT_FALSE(params.model.p_vars.has_value());
  EXPECT_FALSE(params.model.n_vars.has_value());
  params.resolve_defaults(10);
  EXPECT_FLOAT_EQ(params.model.p_vars.value(), 0.5F);
  EXPECT_EQ(params.model.n_vars.value(), 5);
}

/* An explicit p_vars is preserved, n_vars computed from it. */
TEST(ResolveDefaults, ExplicitPVarsPreserved) {
  Params params;
  params.model.mode_input = "classification";
  params.model.size       = 10;
  params.model.p_vars     = 0.8F;
  params.quiet            = true;
  params.resolve_defaults(10);
  EXPECT_FLOAT_EQ(params.model.p_vars.value(), 0.8F);
  EXPECT_EQ(params.model.n_vars.value(), 8);
}

// ---------------------------------------------------------------------------
// resolve_defaults() — seed, threads, and vars auto-resolution
// ---------------------------------------------------------------------------

/* n_vars is computed from p_vars * total_vars. */
TEST(ResolveDefaults, NVarsFromPVars) {
  Params params;
  params.model.mode_input = "classification";
  params.model.size       = 10;
  params.model.p_vars     = 0.5F;
  params.quiet            = true;
  params.resolve_defaults(10);
  EXPECT_EQ(params.model.n_vars.value(), 5);
}

/* p_vars is back-computed from n_vars / total_vars. */
TEST(ResolveDefaults, PVarsFromNVars) {
  Params params;
  params.model.mode_input = "classification";
  params.model.size       = 10;
  params.model.n_vars     = 3;
  params.quiet            = true;
  params.resolve_defaults(10);
  EXPECT_FLOAT_EQ(params.model.p_vars.value(), 0.3F);
}

/* Default vars: p_vars = 0.5, n_vars = half of total. */
TEST(ResolveDefaults, DefaultPVarsAndNVars) {
  Params params;
  params.model.mode_input = "classification";
  params.model.size       = 10;
  params.quiet            = true;
  params.resolve_defaults(10);
  EXPECT_FLOAT_EQ(params.model.p_vars.value(), 0.5F);
  EXPECT_EQ(params.model.n_vars.value(), 5);
}

/* Vars computation is skipped for a single tree (trees = 0). */
TEST(ResolveDefaults, NoVarsWhenSingleTree) {
  Params params;
  params.model.mode_input = "classification";
  params.model.size       = 0;
  params.model.p_vars     = 0.8F;
  params.quiet            = true;
  params.resolve_defaults(10);
  EXPECT_FLOAT_EQ(params.model.p_vars.value(), 0.8F);
  EXPECT_FALSE(params.model.n_vars.has_value());
}

/* Vars computation is skipped when total_vars is 0 (unknown). */
TEST(ResolveDefaults, NoVarsWhenZeroTotalVars) {
  Params params;
  params.model.size = 10;
  params.quiet      = true;
  params.resolve_defaults(0);
  EXPECT_FALSE(params.model.p_vars.has_value());
  EXPECT_FALSE(params.model.n_vars.has_value());
}

/* vars_config with proportion is resolved to count when total_vars is known. */
TEST(ResolveDefaults, VarsProportionResolvedToCount) {
  Params params;
  params.model.mode_input  = "classification";
  params.model.size        = 10;
  params.model.vars_config = {{"name", "uniform"}, {"proportion", 0.5}};
  params.model.p_vars      = 0.5F;
  params.resolve_defaults(10);

  EXPECT_FALSE(params.model.vars_config.contains("proportion"));
  EXPECT_EQ(params.model.vars_config["name"], "uniform");
  EXPECT_EQ(params.model.vars_config["count"].get<int>(), 5);
}

// ---------------------------------------------------------------------------
// warn_unused_params() — single-tree parameter warnings
// ---------------------------------------------------------------------------

/* Single tree with --threads warns that threads is ignored. */
TEST(WarnUnusedParams, TreesZeroThreadsWarning) {
  Params params;
  params.model.size    = 0;
  params.model.threads = 4;
  params.quiet         = false;

  testing::internal::CaptureStdout();
  ppforest2::io::Output out(params.quiet);
  warn_unused_params(out, params);
  std::string output = testing::internal::GetCapturedStdout();

  EXPECT_NE(output.find("threads parameter is ignored"), std::string::npos);
}

/* Single tree with --vars warns that vars is ignored. */
TEST(WarnUnusedParams, TreesZeroVarsWarning) {
  Params params;
  params.model.size   = 0;
  params.model.p_vars = 0.8F;
  params.quiet        = false;

  testing::internal::CaptureStdout();
  ppforest2::io::Output out(params.quiet);
  warn_unused_params(out, params);
  std::string const output = testing::internal::GetCapturedStdout();

  EXPECT_NE(output.find("--n-vars/--p-vars parameter is ignored"), std::string::npos);
}

/* Single tree with both --threads and --vars emits both warnings. */
TEST(WarnUnusedParams, TreesZeroBothWarnings) {
  Params params;
  params.model.size    = 0;
  params.model.threads = 4;
  params.model.n_vars  = 3;
  params.quiet         = false;

  testing::internal::CaptureStdout();
  ppforest2::io::Output out(params.quiet);
  warn_unused_params(out, params);
  std::string const output = testing::internal::GetCapturedStdout();

  EXPECT_NE(output.find("threads parameter is ignored"), std::string::npos);
  EXPECT_NE(output.find("--n-vars/--p-vars parameter is ignored"), std::string::npos);
  EXPECT_NE(output.find("Single trees always use all features"), std::string::npos);
}

/* Forest mode (trees > 0) emits no warnings. */
TEST(WarnUnusedParams, TreesNonZeroNoWarning) {
  Params params;
  params.model.size    = 10;
  params.model.threads = 4;
  params.model.p_vars  = 0.8F;
  params.quiet         = false;

  testing::internal::CaptureStdout();
  ppforest2::io::Output out(params.quiet);
  warn_unused_params(out, params);
  std::string const output = testing::internal::GetCapturedStdout();

  EXPECT_TRUE(output.empty());
}

/* Quiet mode suppresses all parameter warnings. */
TEST(WarnUnusedParams, QuietSuppresses) {
  Params params;
  params.model.size    = 0;
  params.model.threads = 4;
  params.model.p_vars  = 0.8f;
  params.quiet         = true;

  testing::internal::CaptureStdout();
  ppforest2::io::Output out(params.quiet);
  warn_unused_params(out, params);
  std::string const output = testing::internal::GetCapturedStdout();

  EXPECT_TRUE(output.empty());
}


// ---------------------------------------------------------------------------
// parse_args() — additional vars validation (edge cases)
// ---------------------------------------------------------------------------

/* --p-vars of exactly 0.0 must throw — selects no features. */
TEST(ParseArgs, PVarsProportionZeroExits) {
  EXPECT_THROW(parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--p-vars", "0.0"}), ppforest2::UserError);
}

/* --p-vars > 1.0 must throw. */
TEST(ParseArgs, PVarsProportionOutOfRangeExits) {
  EXPECT_THROW(parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--p-vars", "1.5"}), ppforest2::UserError);
}

/* --p-vars with non-numeric string ("abc") must throw. */
TEST(ParseArgs, PVarsInvalidValueExits) {
  EXPECT_THROW(parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--p-vars", "abc"}), ppforest2::UserError);
}

// ---------------------------------------------------------------------------
// parse_args() — explicit strategy flags (--pp, --vars, --cutpoint)
// ---------------------------------------------------------------------------

/* --pp pda without lambda stores the config; validation happens at construction time. */
TEST(ParseArgs, PPStrategyNameOnlyStoresConfig) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--pp", "pda"});
  EXPECT_EQ(opts.model.pp_config["name"], "pda");
}

/* --pp pda:lambda=0.3 sets lambda to 0.3. */
TEST(ParseArgs, PPStrategyWithLambda) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--pp", "pda:lambda=0.3"});
  EXPECT_EQ(opts.model.pp_input, "pda:lambda=0.3");
  EXPECT_FLOAT_EQ(opts.model.pp_config["lambda"].get<float>(), 0.3F);
}

/* --pp and -l are mutually exclusive. */
TEST(ParseArgs, PPExcludesLambda) {
  EXPECT_THROW(
      parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--pp", "pda", "-l", "0.5"}), ppforest2::UserError
  );
}

/* -l and --pp are mutually exclusive (reverse order). */
TEST(ParseArgs, LambdaExcludesPP) {
  EXPECT_THROW(
      parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "-l", "0.5", "--pp", "pda"}), ppforest2::UserError
  );
}

/* --pp with an unknown strategy name stores the config; error at construction time. */
TEST(ParseArgs, PPUnknownStrategyStoresConfig) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--pp", "unknown"});
  EXPECT_EQ(opts.model.pp_config["name"], "unknown");
}

/* --pp pda:unknown=1 stores the config; unknown param rejected at construction time. */
TEST(ParseArgs, PPUnknownParamStoresConfig) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--pp", "pda:unknown=1"});
  EXPECT_EQ(opts.model.pp_config["name"], "pda");
  EXPECT_EQ(opts.model.pp_config["unknown"].get<int>(), 1);
}

/* --pp pda:lambda=abc stores "abc" as a string; type error at construction time. */
TEST(ParseArgs, PPInvalidLambdaValueStoresConfig) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--pp", "pda:lambda=abc"});
  EXPECT_EQ(opts.model.pp_config["lambda"], "abc");
}

/* --pp pda:noequalssign must throw (missing key=value). */
TEST(ParseArgs, PPMissingEqualsExits) {
  EXPECT_THROW(
      parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--pp", "pda:noequalssign"}), ppforest2::UserError
  );
}

/* --vars all sets vars_config without affecting n_vars/p_vars. */
TEST(ParseArgs, VarsStrategyAll) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--vars", "all"});
  EXPECT_EQ(opts.model.vars_config["name"], "all");
  EXPECT_FALSE(opts.model.n_vars.has_value());
  EXPECT_FALSE(opts.model.p_vars.has_value());
}

/* --vars uniform:count=2 sets vars_config without affecting n_vars/p_vars. */
TEST(ParseArgs, VarsStrategyUniformWithCount) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--vars", "uniform:count=2"});
  EXPECT_EQ(opts.model.vars_config["name"], "uniform");
  EXPECT_EQ(opts.model.vars_config["count"].get<int>(), 2);
  EXPECT_FALSE(opts.model.n_vars.has_value());
}

/* --vars uniform:proportion=0.5 sets vars_config without affecting n_vars/p_vars. */
TEST(ParseArgs, VarsStrategyUniformWithProportion) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--vars", "uniform:proportion=0.5"});
  EXPECT_EQ(opts.model.vars_config["name"], "uniform");
  EXPECT_FLOAT_EQ(opts.model.vars_config["proportion"].get<float>(), 0.5F);
  EXPECT_FALSE(opts.model.p_vars.has_value());
}

/* --vars and --n-vars are mutually exclusive. */
TEST(ParseArgs, VarsStrategyExcludesNVars) {
  EXPECT_THROW(
      parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--vars", "all", "--n-vars", "2"}), ppforest2::UserError
  );
}

/* --n-vars and --vars are mutually exclusive (reverse order). */
TEST(ParseArgs, NVarsExcludesVarsStrategy) {
  EXPECT_THROW(
      parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--n-vars", "2", "--vars", "all"}), ppforest2::UserError
  );
}

/* --vars and --p-vars are mutually exclusive. */
TEST(ParseArgs, VarsStrategyExcludesPVars) {
  EXPECT_THROW(
      parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--vars", "all", "--p-vars", "0.5"}), ppforest2::UserError
  );
}

/* --vars with an unknown strategy name stores the config; error at construction time. */
TEST(ParseArgs, VarsStrategyUnknownStoresConfig) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--vars", "unknown"});
  EXPECT_EQ(opts.model.vars_config["name"], "unknown");
}

/* --vars uniform:unknown=1 stores the config; unknown param rejected at construction time. */
TEST(ParseArgs, VarsStrategyUnknownParamStoresConfig) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--vars", "uniform:unknown=1"});
  EXPECT_EQ(opts.model.vars_config["name"], "uniform");
  EXPECT_EQ(opts.model.vars_config["unknown"].get<int>(), 1);
}

/* --cutpoint mean_of_means is accepted. */
TEST(ParseArgs, ThresholdStrategyMeanOfMeans) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--cutpoint", "mean_of_means"});
  EXPECT_EQ(opts.model.cutpoint_input, "mean_of_means");
}

/* --cutpoint with an unknown strategy name stores the config; error at construction time. */
TEST(ParseArgs, ThresholdUnknownStrategyStoresConfig) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--cutpoint", "unknown"});
  EXPECT_EQ(opts.model.cutpoint_config["name"], "unknown");
}

/* --max-retries sets the max retries count. */
TEST(ParseArgs, MaxRetriesOption) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--max-retries", "5"});
  EXPECT_EQ(opts.model.max_retries, 5);
}

/* --max-retries defaults to 3. */
TEST(ParseArgs, MaxRetriesDefault) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str()});
  EXPECT_EQ(opts.model.max_retries, 3);
}

/* All three explicit strategy flags can be used together. */
TEST(ParseArgs, AllExplicitStrategies) {
  auto opts = parse(
      {"ppforest2",
       "train",
       "-d",
       IRIS_PATH.c_str(),
       "--pp",
       "pda:lambda=0.3",
       "--vars",
       "uniform:count=2",
       "--cutpoint",
       "mean_of_means"}
  );
  EXPECT_EQ(opts.model.pp_input, "pda:lambda=0.3");
  EXPECT_EQ(opts.model.vars_input, "uniform:count=2");
  EXPECT_EQ(opts.model.cutpoint_input, "mean_of_means");
  EXPECT_FLOAT_EQ(opts.model.pp_config["lambda"].get<float>(), 0.3F);
  EXPECT_EQ(opts.model.vars_config["count"].get<int>(), 2);
}

/* --pp pda:lambda=0.3 correctly parses single param (baseline for multi-param). */
TEST(ParseArgs, PPStrategySingleParamParsed) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--pp", "pda:lambda=0.3"});
  EXPECT_EQ(opts.model.pp_config["name"], "pda");
  EXPECT_FLOAT_EQ(opts.model.pp_config["lambda"].get<float>(), 0.3F);
  EXPECT_EQ(opts.model.pp_config.size(), 2u); // name + lambda only
}

/* --pp pda:lambda=0.3,unknown=1 stores all params; unknown rejected at construction time. */
TEST(ParseArgs, PPStrategyMultipleParamsStoresAll) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--pp", "pda:lambda=0.3,unknown=1"});
  EXPECT_FLOAT_EQ(opts.model.pp_config["lambda"].get<float>(), 0.3F);
  EXPECT_EQ(opts.model.pp_config["unknown"].get<int>(), 1);
}

/* --vars uniform:count=2,unknown=1 stores all params; unknown rejected at construction time. */
TEST(ParseArgs, VarsStrategyMultipleParamsStoresAll) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--vars", "uniform:count=2,unknown=1"});
  EXPECT_EQ(opts.model.vars_config["count"].get<int>(), 2);
  EXPECT_EQ(opts.model.vars_config["unknown"].get<int>(), 1);
}

// ---------------------------------------------------------------------------
// --stop (repeatable; wraps in any(...) when > 1)
// ---------------------------------------------------------------------------

/* Single --stop parses exactly like the pre-existing flat syntax — no
 * CompositeStop wrapping, so saved models built on a single --stop have the
 * same JSON shape as before the repeat-enabled change. */
TEST(ParseArgs, StopSingleOccurrenceNoCompositeWrap) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--stop", "pure_node"});
  EXPECT_EQ(opts.model.stop_config["name"], "pure_node");
  EXPECT_FALSE(opts.model.stop_config.contains("rules"));
}

/* Single --stop with params round-trips name + params into stop_config. */
TEST(ParseArgs, StopSingleOccurrenceWithParam) {
  auto opts = parse({"ppforest2", "train", "-d", IRIS_PATH.c_str(), "--stop", "min_size:min_size=5"});
  EXPECT_EQ(opts.model.stop_config["name"], "min_size");
  EXPECT_EQ(opts.model.stop_config["min_size"].get<int>(), 5);
  EXPECT_FALSE(opts.model.stop_config.contains("rules"));
}

/* Two --stop occurrences compose into an `any` CompositeStop. This is the
 * CLI-ergonomic path for the regression default: users who'd otherwise
 * have to write a full --config JSON can now tweak the composite inline. */
TEST(ParseArgs, StopRepeatedWrapsInAny) {
  auto opts = parse({
      "ppforest2",
      "train",
      "-d",
      IRIS_PATH.c_str(),
      "--stop",
      "min_size:min_size=5",
      "--stop",
      "min_variance:threshold=0.01",
  });
  EXPECT_EQ(opts.model.stop_config["name"], "any");
  ASSERT_TRUE(opts.model.stop_config.contains("rules"));
  ASSERT_EQ(opts.model.stop_config["rules"].size(), 2U);
  EXPECT_EQ(opts.model.stop_config["rules"][0]["name"], "min_size");
  EXPECT_EQ(opts.model.stop_config["rules"][0]["min_size"].get<int>(), 5);
  EXPECT_EQ(opts.model.stop_config["rules"][1]["name"], "min_variance");
  EXPECT_FLOAT_EQ(opts.model.stop_config["rules"][1]["threshold"].get<float>(), 0.01F);
}

/* Three --stop occurrences: rules array preserves order. */
TEST(ParseArgs, StopRepeatedPreservesOrder) {
  auto opts = parse({
      "ppforest2",
      "train",
      "-d",
      IRIS_PATH.c_str(),
      "--stop",
      "pure_node",
      "--stop",
      "min_size:min_size=3",
      "--stop",
      "min_variance:threshold=0.02",
  });
  EXPECT_EQ(opts.model.stop_config["name"], "any");
  ASSERT_EQ(opts.model.stop_config["rules"].size(), 3U);
  EXPECT_EQ(opts.model.stop_config["rules"][0]["name"], "pure_node");
  EXPECT_EQ(opts.model.stop_config["rules"][1]["name"], "min_size");
  EXPECT_EQ(opts.model.stop_config["rules"][2]["name"], "min_variance");
}

// ---------------------------------------------------------------------------
// summarize subcommand parsing
// ---------------------------------------------------------------------------

/* Parsing "summarize" selects the summarize subcommand. */
TEST(ParseArgs, SummarizeSubcommand) {
  auto opts = parse({"ppforest2", "summarize", "-M", IRIS_PATH.c_str()});
  EXPECT_EQ(opts.subcommand, Subcommand::summarize);
  EXPECT_EQ(opts.model_path, IRIS_PATH);
}

/* Summarize without required -M must throw. */
TEST(ParseArgs, SummarizeWithoutModelExits) {
  EXPECT_THROW(parse({"ppforest2", "summarize"}), ppforest2::UserError);
}
