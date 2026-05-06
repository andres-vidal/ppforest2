/**
 * @file Evaluate.cpp
 * @brief Multi-iteration model evaluation with convergence and the
 *        evaluate subcommand handler.
 */
#include "cli/Evaluate.hpp"
#include "cli/Train.hpp"
#include "cli/Validation.hpp"

#include <CLI/CLI.hpp>
#include <fmt/format.h>
#include <filesystem>
#include <Eigen/Dense>

#include "models/Evaluation.hpp"
#include "stats/Simulation.hpp"
#include "io/Presentation.hpp"
#include "utils/System.hpp"
#include "io/Color.hpp"
#include "io/IO.hpp"
#include "utils/UserError.hpp"
#include "io/Output.hpp"

#include <nlohmann/json.hpp>

using namespace ppforest2;
using namespace ppforest2::types;
using namespace ppforest2::stats;
using namespace ppforest2::io::style;
using json = nlohmann::json;

namespace ppforest2::cli {
  namespace {
    void set_sd(SimulateParams& simulation, float v) {
      simulation.classification.sd = v;
      simulation.regression.sd     = v;
    }

    void add_simulation_options(CLI::App* sub, SimulateParams& simulation) {
      auto& cls = simulation.classification;
      auto& reg = simulation.regression;

      sub->add_option("--simulate", simulation.format, "Simulate NxPxG data");
      sub->add_option("--simulate-mean", cls.mean, "Base mean for the first group (default: 100.0)");
      sub->add_option("--simulate-mean-separation", cls.mean_separation, "Mean shift between groups (default: 50.0)");
      sub->add_option("--simulate-n-informative", reg.n_informative, "Number of features driving the response");
      sub->add_option("--simulate-y-intercept", reg.y_intercept, "Response intercept (default: 0.0)");
      sub->add_option("--simulate-y-sd", reg.y_sd, "Response noise SD (default: 0.1)");

      sub->add_option_function<float>(
          "--simulate-sd",
          [&simulation](float v) { set_sd(simulation, v); },
          "Feature standard deviation (classification default: 10.0, regression default: 1.0)"
      );

      sub->get_option("--simulate-mean")->needs("--simulate");
      sub->get_option("--simulate-mean-separation")->needs("--simulate");
      sub->get_option("--simulate-n-informative")->needs("--simulate");
      sub->get_option("--simulate-y-intercept")->needs("--simulate");
      sub->get_option("--simulate-y-sd")->needs("--simulate");
      sub->get_option("--simulate-sd")->needs("--simulate");
    }
  }


  void add_evaluate_options(CLI::App* sub, EvaluateParams& evaluate) {

    auto& convergence = evaluate.convergence;

    sub->add_option("-p,--train-ratio", evaluate.train_ratio, "Train set ratio (default: 0.7)");
    sub->add_option("-i,--iterations", evaluate.iterations, "Fixed iteration count (disables convergence)");
    sub->add_option("--warmup", evaluate.warmup, "Warmup iterations to discard before measuring (default: 0)");
    sub->add_flag(
        "--fixed-seed",
        evaluate.fixed_seed,
        "Use the same seed for every iteration (pure timing-noise measurement; std_*_error will be 0). "
        "Default: each iteration draws a fresh seed for genuine model variance."
    );
    sub->add_option("--convergence-cv", convergence.cv, "CV threshold for convergence (default: 0.05)");
    sub->add_option("--convergence-min", convergence.min, "Min iterations before checking convergence (default: 10)");
    sub->add_option("--convergence-max", convergence.max, "Max iterations for convergence (default: 200)");
    sub->add_option("--convergence-window", convergence.window, "Consecutive stable checks to stop (default: 3)");

    sub->get_option("--iterations")->excludes("--convergence-cv");
    sub->get_option("--iterations")->excludes("--convergence-window");
    sub->get_option("--iterations")->excludes("--convergence-min");
    sub->get_option("--iterations")->excludes("--convergence-max");
  }

  void setup_evaluate(CLI::App& app, Params& params) {
    auto* sub = app.add_subcommand("evaluate", "Train and evaluate a model");
    sub->add_option("-d,--data", params.data_path, "CSV file");

    add_model_options(sub, params.model);
    add_evaluate_options(sub, params.evaluate);
    add_simulation_options(sub, params.simulation);

    sub->add_option("-o,--output", params.output_path, "Save evaluation results to JSON file");
    sub->add_option("-e,--export", params.evaluate.export_path, "Export experiment bundle to directory");

    // CLI-exclusive constraints (data/simulate mutual exclusion,
    // simulate sub-params need simulate)
    sub->get_option("--data")->excludes("--simulate");
    sub->get_option("--simulate")->excludes("--data");
    ;

    sub->callback([&]() { params.subcommand = Subcommand::evaluate; });
  }

  namespace {
    template<typename T> Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1> const> as_vector(std::vector<T> const& v) {
      return {v.data(), static_cast<Eigen::Index>(v.size())};
    }

    template<typename Derived>
    bool check_convergence(Eigen::MatrixBase<Derived> const& values, EvaluateParams::Convergence const& conv) {
      if (values.size() < *conv.min) {
        return false;
      }

      double mean = static_cast<double>(values.mean());

      if (mean <= 0) {
        return true;
      }

      double cv = stats::sd(values) / mean;

      return cv < *conv.cv;
    }

    /**
     * @brief Tracks convergence state across iterations.
     *
     * In fixed-iteration mode, does nothing. In convergence mode,
     * tracks how many consecutive iterations all metrics are stable.
     */
    struct ConvergenceTracker {
      EvaluateParams const& evaluate;

      int max_iters;
      int stable_count = 0;

      explicit ConvergenceTracker(EvaluateParams const& evaluate)
          : evaluate(evaluate)
          , max_iters(evaluate.convergence_enabled() ? *evaluate.convergence.max : *evaluate.iterations) {}

      bool enabled() const { return evaluate.convergence_enabled(); }

      void print_header(io::Output& out) const {
        if (enabled()) {
          out.println(
              "Running up to {} iterations (converge at CV < {:.0f}%):",
              emphasis(std::to_string(max_iters)),
              evaluate.convergence.cv.value() * 100
          );
        } else {
          out.println("Running {} iterations:", emphasis(std::to_string(max_iters)));
        }
      }

      template<typename T> bool converged(std::vector<T> const& values) const {
        return check_convergence(as_vector(values), evaluate.convergence);
      }

      bool update(
          std::vector<long long> const& times,
          std::vector<double> const& tr_errors,
          std::vector<double> const& te_errors
      ) {
        if (!enabled()) {
          return false;
        }

        if (converged(times) && converged(tr_errors) && converged(te_errors)) {
          stable_count++;
          return stable_count >= *evaluate.convergence.window;
        }

        stable_count = 0;
        return false;
      }

      void print_result(io::Output& out, int iterations_run) const {
        out.newline();

        if (!enabled()) {
          return;
        }

        if (stable_count >= *evaluate.convergence.window) {
          out.println("Converged after {} iterations (CV < {:.0f}%)", iterations_run, *evaluate.convergence.cv * 100);
        } else {
          out.println("{} Did not converge after {} iterations", warning("Warning:"), iterations_run);
          out.newline();
        }
      }
    };

    io::ModelStats evaluate_model(
        io::Output& out,
        FeatureMatrix const& tr_x,
        FeatureMatrix const& te_x,
        OutcomeVector const& tr_y,
        OutcomeVector const& te_y,
        Params const& params,
        ppforest2::stats::RNG& rng
    ) {
      // Default: each iteration trains with a fresh seed drawn from `rng`
      // so models actually differ across iterations — making the
      // train/test error variance measurements meaningful (rather than
      // reading constant zeros from re-using the same seed).
      //
      // `--fixed-seed`: every iteration uses `params.model.seed` unchanged.
      // Models are identical, `std_*_error` is tautologically zero, and
      // `std_time_ms` reflects pure system noise. Useful when benchmarking
      // the timing stability of one specific configuration.
      //
      // The seed mutates on a single local copy (kept across iterations)
      // so the caller's `params` stays untouched and we avoid re-copying
      // the whole struct per iteration.
      Params iter_params   = params;
      bool const vary_seed = !params.evaluate.fixed_seed;

      int warmup = params.evaluate.warmup;

      // Run warmup iterations (discarded)
      if (warmup > 0) {
        out.println("Running {} warmup iterations:", emphasis(std::to_string(warmup)));
      }

      for (int i = 0; i < warmup; ++i) {
        out.progress(i, warmup);
        if (vary_seed) {
          iter_params.model.seed = static_cast<int>(rng());
        }
        train_model(tr_x, tr_y, iter_params);
        out.progress(i + 1, warmup);
      }

      if (warmup > 0) {
        out.newline();
      }

      // Measured iterations
      ConvergenceTracker tracker(params.evaluate);
      tracker.print_header(out);

      std::vector<long long> times;
      std::vector<double> tr_errors;
      std::vector<double> te_errors;

      for (int i = 0; i < tracker.max_iters; ++i) {
        out.progress(i, tracker.max_iters);

        if (vary_seed) {
          iter_params.model.seed = static_cast<int>(rng());
        }
        auto const train_result = train_model(tr_x, tr_y, iter_params);

        times.push_back(train_result.duration);
        tr_errors.push_back(error(*train_result.model, tr_x, tr_y));
        te_errors.push_back(error(*train_result.model, te_x, te_y));

        out.progress(i + 1, tracker.max_iters);

        if (tracker.update(times, tr_errors, te_errors)) {
          int done = static_cast<int>(times.size());
          out.progress(done, done);
          break;
        }
      }

      tracker.print_result(out, static_cast<int>(times.size()));

      io::ModelStats stats;
      stats.tr_times = as_vector(times);
      stats.tr_error = as_vector(tr_errors);
      stats.te_error = as_vector(te_errors);

      return stats;
    }
  }

  int run_evaluate(Params& params) {
    // Snapshot which params are unset before defaults are filled
    bool default_seed    = !params.model.seed;
    bool default_threads = !params.model.threads;
    bool default_vars    = !params.model.p_vars && !params.model.n_vars;

    params.evaluate.resolve_defaults();
    params.resolve_seed();
    validate_params(params);

    if (!params.output_path.empty()) {
      io::check_file_not_exists(params.output_path);
    }

    if (!params.evaluate.export_path.empty()) {
      io::check_dir_not_exists(params.evaluate.export_path);
    }

    ppforest2::stats::RNG rng(*params.model.seed);
    auto full_data = params.read_data(rng);

    params.resolve_defaults(full_data.x.cols());

    auto data_split = split(full_data, *params.evaluate.train_ratio, rng);

    FeatureMatrix tr_x = full_data.x(data_split.tr, Eigen::all);
    FeatureMatrix te_x = full_data.x(data_split.te, Eigen::all);
    OutcomeVector tr_y = full_data.y(data_split.tr);
    OutcomeVector te_y = full_data.y(data_split.te);

    io::Output out(params.quiet);
    warn_unused_params(out, params);

    {
      json const config = params.model.to_json();

      io::ConfigDisplayHints hints;
      hints.vars_percent     = params.model.size > 0 && params.model.p_vars ? params.model.p_vars.value() * 100 : -1;
      hints.default_vars     = default_vars;
      hints.default_threads  = default_threads;
      hints.default_seed     = default_seed;
      hints.training_samples = fmt::format("{} ({}%)", tr_x.rows(), (*params.evaluate.train_ratio * 100));
      hints.test_samples     = fmt::format("{} ({}%)", te_x.rows(), (1 - *params.evaluate.train_ratio) * 100);

      io::print_configuration(out, config, hints);
    }

    {
      json meta;
      meta["observations"] = static_cast<int>(full_data.x.rows());
      meta["features"]     = static_cast<int>(full_data.x.cols());

      if (!full_data.group_names.empty()) {
        meta["groups"] = full_data.group_names;
      }

      io::print_data_summary(out, meta);
    }

    auto stats = evaluate_model(out, tr_x, te_x, tr_y, te_y, params, rng);

    // Data and model info for downstream consumers (e.g. benchmark)
    stats.data_path   = params.data_path;
    stats.mode        = types::mode_from_string(params.model.mode_input);
    stats.n           = static_cast<int>(full_data.x.rows());
    stats.p           = static_cast<int>(full_data.x.cols());
    stats.g           = static_cast<int>(full_data.groups.size());
    stats.size        = params.model.size;
    stats.n_vars      = params.model.n_vars;
    stats.p_vars      = params.model.p_vars;
    stats.train_ratio = *params.evaluate.train_ratio;

    // Measure peak RSS after evaluation
    stats.peak_rss_bytes = ppforest2::sys::get_peak_rss_bytes();

    print_results(out, stats);

    // Save results to file if requested
    if (!params.output_path.empty()) {
      io::json::write_file(stats.to_json(), params.output_path, user_error);
      out.saved("Results", params.output_path);
    }

    // Export experiment bundle if requested
    if (!params.evaluate.export_path.empty()) {
      std::string dir = params.evaluate.export_path;
      std::filesystem::create_directories(dir);

      json config    = params.to_json();
      config["data"] = "data.csv";
      io::json::write_file(config, dir + "/config.json", user_error);
      io::json::write_file(stats.to_json(), dir + "/results.json", user_error);
      io::csv::write(full_data, dir + "/data.csv");

      out.println("{}{}/", success("Experiment exported to "), dir);
    }

    return 0;
  }
}
