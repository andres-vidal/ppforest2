/**
 * @file ServeHandlers.test.cpp
 * @brief Unit tests for the pure handler functions backing `ppforest2 serve`.
 *
 * Exercises (request → response) transformation without bringing up an HTTP
 * server. Models are produced via the binary's `train` subcommand at fixture
 * setup time, then loaded in-process from the saved JSON.
 */
#include "cli/CLI.integration.hpp"
#include "cli/ServeHandlers.hpp"
#include "io/IO.hpp"
#include "serialization/Json.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <regex>
#include <string>

using json = nlohmann::json;
using ppforest2::cli::serve::handle_health;
using ppforest2::cli::serve::handle_predict;
using ppforest2::cli::serve::handle_predict_html;
using ppforest2::cli::serve::handle_predict_view;
using ppforest2::cli::serve::handle_summarize;
using ppforest2::cli::serve::handle_summary_html;
using ppforest2::cli::serve::LoadedModel;
using ppforest2::cli::serve::PredictionResult;
using ppforest2::cli::serve::PredictionStore;
using ppforest2::cli::serve::Response;

namespace {
  LoadedModel load_from_file(std::string const& path) {
    json model_json = ppforest2::io::json::read_file(path);
    auto exported   = model_json.get<ppforest2::serialization::Export<ppforest2::Model::Ptr>>();
    auto mode       = exported.spec->mode;

    return LoadedModel{
        exported.model,
        std::move(exported.groups),
        std::move(exported.feature_names),
        mode,
        std::move(model_json),
        exported.n_features,
    };
  }
}

class ServeHandlersClassificationTest : public SavedModelTest {
protected:
  PredictionStore store;
};

TEST_F(ServeHandlersClassificationTest, HealthReturnsOkWithVersion) {
  Response const r = handle_health("9.9.9");

  EXPECT_EQ(r.status, 200);
  auto body = json::parse(r.body);
  EXPECT_EQ(body.at("status"), "ok");
  EXPECT_EQ(body.at("version"), "9.9.9");
}

/* Models saved without feature names (e.g. via the R bindings) cannot be
 * permuted by name: the request must carry exactly the training feature
 * count. A mismatched count must be a 400 — it used to slice a narrower
 * matrix and read out of bounds, crashing the server. */
TEST_F(ServeHandlersClassificationTest, PredictUnnamedModelWrongColumnCountIs400) {
  auto mj                     = json::parse(model_->read());
  mj["meta"]["feature_names"] = json::array();
  TempFile const stripped;
  {
    std::ofstream out(stripped.path());
    out << mj.dump();
  }
  auto const loaded = load_from_file(stripped.path());

  // One column instead of four.
  Response const r = handle_predict(loaded, store, "f1\n5.1\n");
  EXPECT_EQ(r.status, 400);
  EXPECT_NE(r.body.find("Feature mismatch"), std::string::npos);
}

/* With the exact training feature count, an unnamed model predicts
 * positionally regardless of the header names. */
TEST_F(ServeHandlersClassificationTest, PredictUnnamedModelPositionalColumnsWork) {
  auto mj                     = json::parse(model_->read());
  mj["meta"]["feature_names"] = json::array();
  TempFile const stripped;
  {
    std::ofstream out(stripped.path());
    out << mj.dump();
  }
  auto const loaded = load_from_file(stripped.path());

  Response const r = handle_predict(loaded, store, "a,b,c,d\n5.1,3.5,1.4,0.2\n");
  EXPECT_EQ(r.status, 200);
  auto body = json::parse(r.body);
  EXPECT_EQ(body.at("predictions").size(), 1U);
}

TEST_F(ServeHandlersClassificationTest, SummarizeOmitsModelField) {
  auto const loaded = load_from_file(model_->path());

  Response const r = handle_summarize(loaded);

  EXPECT_EQ(r.status, 200);
  EXPECT_EQ(r.content_type, "application/json");
  auto body = json::parse(r.body);
  EXPECT_FALSE(body.contains("model"));
  EXPECT_TRUE(body.contains("meta"));
  EXPECT_TRUE(body.contains("config"));
  EXPECT_EQ(body.at("meta").at("feature_names").size(), 4u);
}

TEST_F(ServeHandlersClassificationTest, SummaryHtmlIncludesKeyFields) {
  auto const loaded = load_from_file(model_->path());

  Response const r = handle_summary_html(loaded);

  EXPECT_EQ(r.status, 200);
  EXPECT_NE(r.content_type.find("text/html"), std::string::npos);
  EXPECT_EQ(r.body.find("<?xml"), std::string::npos);
  EXPECT_NE(r.body.find("<!DOCTYPE html>"), std::string::npos);
  EXPECT_NE(r.body.find("Classification"), std::string::npos);
  EXPECT_NE(r.body.find("Petal.Length"), std::string::npos);
  EXPECT_NE(r.body.find("Accuracy"), std::string::npos);
  EXPECT_NE(r.body.find("Variable importance"), std::string::npos);
  EXPECT_EQ(r.body.find("Regression"), std::string::npos);
}

TEST_F(ServeHandlersClassificationTest, SummaryHtmlEmbedsStyleAndScript) {
  auto const loaded = load_from_file(model_->path());

  Response const r = handle_summary_html(loaded);

  // Style and script tags must actually contain content. A formatter that
  // mangles `{{ style }}` / `{{ script }}` inside <style>/<script> would
  // leave empty / broken bodies, breaking the dashboard silently — these
  // assertions catch that regression.
  EXPECT_NE(r.body.find("<style>"), std::string::npos);
  EXPECT_NE(r.body.find(".card"), std::string::npos) << "CSS not embedded in <style>";
  EXPECT_NE(r.body.find("<script>"), std::string::npos);
  EXPECT_NE(r.body.find("addEventListener"), std::string::npos) << "JS not embedded in <script>";
}

TEST_F(ServeHandlersClassificationTest, SummaryHtmlShowsConfigSection) {
  auto const loaded = load_from_file(model_->path());

  Response const r = handle_summary_html(loaded);

  EXPECT_NE(r.body.find("Configuration"), std::string::npos);
  EXPECT_NE(r.body.find("Projection pursuit"), std::string::npos);
  EXPECT_NE(r.body.find("Variable selection"), std::string::npos);
  EXPECT_NE(r.body.find("Forest size"), std::string::npos);
  EXPECT_NE(r.body.find("Seed"), std::string::npos);
  // The model JSON's `data` and `mode` keys are intentionally not surfaced;
  // mode is in the header and data is the source CSV path.
  EXPECT_EQ(r.body.find("Mode</dt>"), std::string::npos);
}

TEST_F(ServeHandlersClassificationTest, SummaryHtmlIncludesForestSize) {
  auto const loaded = load_from_file(model_->path());

  Response const r = handle_summary_html(loaded);

  // SavedModelTest trains with `-n 5` (5 trees), so the subtitle should
  // surface the forest size.
  EXPECT_NE(r.body.find("Forest of 5 trees"), std::string::npos);
}

TEST_F(ServeHandlersClassificationTest, PredictPagesShowModelIdentifier) {
  auto const loaded = load_from_file(model_->path());

  Response const empty = handle_predict_view(loaded, store, "");

  // Empty predict landing page surfaces the same identifier as the summary
  // subtitle — so users know which model they're predicting against.
  EXPECT_NE(empty.body.find("Forest of 5 trees"), std::string::npos);
  EXPECT_NE(empty.body.find("Classification"), std::string::npos);

  std::string const csv    = "Sepal.Length,Sepal.Width,Petal.Length,Petal.Width\n"
                             "5.1,3.5,1.4,0.2\n";
  Response const populated = handle_predict_html(loaded, store, csv);
  EXPECT_NE(populated.body.find("Forest of 5 trees"), std::string::npos);
}

TEST_F(ServeHandlersClassificationTest, SummaryHtmlShowsStrategyDisplayNames) {
  // Train with an explicit fractional lambda so the compaction assertion below
  // exercises trailing-zero trimming independent of the default lambda.
  TempFile model_file;
  model_file.clear();
  auto train = run_ppforest2("-q train -d " + IRIS_CSV + " -n 5 -r 0 -l 0.5 -s " + model_file.path());
  ASSERT_EQ(train.exit_code, 0);

  auto const loaded = load_from_file(model_file.path());

  Response const r = handle_summary_html(loaded);

  // Display-name conversions, not the raw snake_case identifiers.
  EXPECT_NE(r.body.find("PDA"), std::string::npos);
  EXPECT_NE(r.body.find("Mean of means"), std::string::npos);
  EXPECT_NE(r.body.find("Pure node"), std::string::npos);
  EXPECT_EQ(r.body.find(">pda<"), std::string::npos);
  EXPECT_EQ(r.body.find("mean_of_means"), std::string::npos);

  // Lambda compacts trailing zeros, no 4-digit padding.
  EXPECT_NE(r.body.find("lambda=0.5"), std::string::npos);
  EXPECT_EQ(r.body.find("lambda=0.5000"), std::string::npos);
}

TEST_F(ServeHandlersClassificationTest, PredictHtmlHasPredictionsTable) {
  auto const loaded = load_from_file(model_->path());

  std::string const csv = "Sepal.Length,Sepal.Width,Petal.Length,Petal.Width\n"
                          "5.1,3.5,1.4,0.2\n"
                          "6.7,3.0,5.2,2.3\n";

  Response const r = handle_predict_html(loaded, store, csv);

  ASSERT_EQ(r.status, 200);
  EXPECT_NE(r.content_type.find("text/html"), std::string::npos);
  EXPECT_NE(r.body.find("<!DOCTYPE html>"), std::string::npos);
  EXPECT_NE(r.body.find("<table class=\"predictions\""), std::string::npos);
  EXPECT_NE(r.body.find("Prediction"), std::string::npos);
  // Two data rows in the body. The whitespace inside `<td>` cells comes from
  // the XHTML template (which the formatter re-indents), so we look for the
  // row index between any whitespace.
  EXPECT_TRUE(std::regex_search(r.body, std::regex(R"(<td>\s*1\s*</td>)")));
  EXPECT_TRUE(std::regex_search(r.body, std::regex(R"(<td>\s*2\s*</td>)")));
  // No leftover ppf: tags.
  EXPECT_EQ(r.body.find("<ppf:"), std::string::npos);
}

TEST_F(ServeHandlersClassificationTest, PredictHtmlIncludesProportionHeaders) {
  auto const loaded = load_from_file(model_->path());

  std::string const csv = "Sepal.Length,Sepal.Width,Petal.Length,Petal.Width\n"
                          "5.1,3.5,1.4,0.2\n";

  Response const r = handle_predict_html(loaded, store, csv);

  for (auto const& name : loaded.group_names) {
    EXPECT_NE(r.body.find(name), std::string::npos) << "missing class header: " << name;
  }
}

TEST_F(ServeHandlersClassificationTest, PredictHtmlIncludesDownloadButton) {
  auto const loaded = load_from_file(model_->path());

  std::string const csv = "Sepal.Length,Sepal.Width,Petal.Length,Petal.Width\n"
                          "5.1,3.5,1.4,0.2\n";

  Response const r = handle_predict_html(loaded, store, csv);

  EXPECT_NE(r.body.find("id=\"download-csv\""), std::string::npos);
}

TEST_F(ServeHandlersClassificationTest, PredictHtmlShowsMetricsWhenLabelsPresent) {
  auto const loaded = load_from_file(model_->path());

  // Request CSV includes the original "Type" label column. Server should
  // ignore it for the feature matrix but use it as ground truth for metrics.
  std::string const csv = "Sepal.Length,Sepal.Width,Petal.Length,Petal.Width,Type\n"
                          "5.1,3.5,1.4,0.2,setosa\n"
                          "6.7,3.0,5.2,2.3,virginica\n";

  Response const r = handle_predict_html(loaded, store, csv);

  ASSERT_EQ(r.status, 200);
  EXPECT_NE(r.body.find("Metrics"), std::string::npos);
  EXPECT_NE(r.body.find("Accuracy"), std::string::npos);
}

TEST_F(ServeHandlersClassificationTest, PredictHtmlShowsConfusionMatrixWhenLabelsPresent) {
  auto const loaded = load_from_file(model_->path());

  std::string const csv = "Sepal.Length,Sepal.Width,Petal.Length,Petal.Width,Type\n"
                          "5.1,3.5,1.4,0.2,setosa\n"
                          "6.7,3.0,5.2,2.3,virginica\n";

  Response const r = handle_predict_html(loaded, store, csv);

  ASSERT_EQ(r.status, 200);
  EXPECT_NE(r.body.find("Confusion matrix"), std::string::npos);
  EXPECT_NE(r.body.find("table class=\"confusion\""), std::string::npos);
  // Class names appear both as column headers and row labels.
  for (auto const& name : loaded.group_names) {
    EXPECT_NE(r.body.find(name), std::string::npos);
  }
  // Green/red/muted colour-coded cells (matching the CLI's confusion matrix).
  EXPECT_NE(r.body.find("class=\"cell correct\""), std::string::npos);
}

TEST_F(ServeHandlersClassificationTest, PredictHtmlOmitsConfusionMatrixWithoutLabels) {
  auto const loaded = load_from_file(model_->path());

  std::string const csv = "Sepal.Length,Sepal.Width,Petal.Length,Petal.Width\n"
                          "5.1,3.5,1.4,0.2\n";

  Response const r = handle_predict_html(loaded, store, csv);

  EXPECT_EQ(r.body.find("Confusion matrix"), std::string::npos);
  EXPECT_EQ(r.body.find("table class=\"confusion\""), std::string::npos);
}

TEST_F(ServeHandlersClassificationTest, PredictHtmlOmitsMetricsWithoutLabels) {
  auto const loaded = load_from_file(model_->path());

  std::string const csv = "Sepal.Length,Sepal.Width,Petal.Length,Petal.Width\n"
                          "5.1,3.5,1.4,0.2\n";

  Response const r = handle_predict_html(loaded, store, csv);

  EXPECT_EQ(r.body.find("Metrics</h2>"), std::string::npos);
  EXPECT_EQ(r.body.find("Accuracy"), std::string::npos);
}

TEST_F(ServeHandlersClassificationTest, PredictJsonIncludesMetricsWhenLabelsPresent) {
  auto const loaded = load_from_file(model_->path());

  std::string const csv = "Sepal.Length,Sepal.Width,Petal.Length,Petal.Width,Type\n"
                          "5.1,3.5,1.4,0.2,setosa\n"
                          "6.7,3.0,5.2,2.3,virginica\n";

  Response const r = handle_predict(loaded, store, csv);

  ASSERT_EQ(r.status, 200);
  auto body = json::parse(r.body);
  ASSERT_TRUE(body.contains("metrics"));
  EXPECT_TRUE(body.at("metrics").contains("error_rate"));
}

TEST_F(ServeHandlersClassificationTest, PredictHtmlSetsContentLocationToShareableUrl) {
  auto const loaded = load_from_file(model_->path());

  std::string const csv = "Sepal.Length,Sepal.Width,Petal.Length,Petal.Width\n"
                          "5.1,3.5,1.4,0.2\n";

  Response const r = handle_predict_html(loaded, store, csv);

  ASSERT_EQ(r.status, 200);
  auto const it = r.headers.find("Content-Location");
  ASSERT_NE(it, r.headers.end()) << "expected Content-Location header on POST /predict";
  EXPECT_NE(it->second.find("/predict?id="), std::string::npos);
}

TEST_F(ServeHandlersClassificationTest, PredictViewReturnsCachedResultById) {
  auto const loaded = load_from_file(model_->path());

  std::string const csv = "Sepal.Length,Sepal.Width,Petal.Length,Petal.Width\n"
                          "5.1,3.5,1.4,0.2\n";

  Response const posted = handle_predict_html(loaded, store, csv);
  auto const it         = posted.headers.find("Content-Location");
  ASSERT_NE(it, posted.headers.end());
  std::string const url = it->second;
  // Extract the id from "/predict?id=<id>".
  std::string const id = url.substr(url.find('=') + 1);

  Response const got = handle_predict_view(loaded, store, id);

  EXPECT_EQ(got.status, 200);
  EXPECT_NE(got.body.find("<table class=\"predictions\""), std::string::npos);
  EXPECT_NE(got.headers.find("Content-Location"), got.headers.end());
}

TEST_F(ServeHandlersClassificationTest, PredictViewShowsFormWithoutId) {
  auto const loaded = load_from_file(model_->path());

  Response const r = handle_predict_view(loaded, store, "");

  EXPECT_EQ(r.status, 200);
  EXPECT_NE(r.body.find("id=\"predict-form\""), std::string::npos);
  EXPECT_EQ(r.body.find("<table class=\"predictions\""), std::string::npos);
  // No Content-Location when there's nothing canonical to point at.
  EXPECT_EQ(r.headers.find("Content-Location"), r.headers.end());
}

TEST_F(ServeHandlersClassificationTest, PredictViewWithUnknownIdShowsFormOnly) {
  auto const loaded = load_from_file(model_->path());

  Response const r = handle_predict_view(loaded, store, "deadbeef00000000");

  EXPECT_EQ(r.status, 200);
  EXPECT_NE(r.body.find("id=\"predict-form\""), std::string::npos);
  EXPECT_EQ(r.body.find("<table class=\"predictions\""), std::string::npos);
}

TEST_F(ServeHandlersClassificationTest, NavLinksPresentOnBothPages) {
  auto const loaded = load_from_file(model_->path());

  std::string const summary = handle_summary_html(loaded).body;
  std::string const predict = handle_predict_view(loaded, store, "").body;

  for (auto const& page : {std::cref(summary), std::cref(predict)}) {
    EXPECT_NE(page.get().find("href=\"/\""), std::string::npos);
    EXPECT_NE(page.get().find("href=\"/predict\""), std::string::npos);
  }
}

TEST(PredictionStoreCapacityTest, EvictsOldestWhenFull) {
  PredictionStore store(2);
  PredictionResult a;
  PredictionResult b;
  PredictionResult c;

  auto const id_a = store.put(std::move(a));
  auto const id_b = store.put(std::move(b));
  EXPECT_TRUE(store.get(id_a).has_value());
  EXPECT_TRUE(store.get(id_b).has_value());

  auto const id_c = store.put(std::move(c));
  EXPECT_FALSE(store.get(id_a).has_value()) << "oldest entry should have been evicted";
  EXPECT_TRUE(store.get(id_b).has_value());
  EXPECT_TRUE(store.get(id_c).has_value());
}

TEST_F(ServeHandlersClassificationTest, PredictHtmlReturnsJsonErrorOnBadInput) {
  auto const loaded = load_from_file(model_->path());

  Response const r = handle_predict_html(loaded, store, "Sepal.Length\n5.1\n");

  EXPECT_EQ(r.status, 400);
  EXPECT_EQ(r.content_type, "application/json");
}

TEST_F(ServeHandlersClassificationTest, SummaryHtmlHasFileInputForPredict) {
  auto const loaded = load_from_file(model_->path());

  Response const r = handle_summary_html(loaded);

  // The formatter spreads <input> attributes across lines, so match the tag
  // and the `type="file"` attribute independently rather than as one literal.
  EXPECT_NE(r.body.find("<input"), std::string::npos);
  EXPECT_NE(r.body.find("type=\"file\""), std::string::npos);
  EXPECT_NE(r.body.find("id=\"predict-form\""), std::string::npos);
  EXPECT_NE(r.body.find("/predict"), std::string::npos);
}

TEST_F(ServeHandlersClassificationTest, SummaryHtmlStripsAllPpfTags) {
  auto const loaded = load_from_file(model_->path());

  Response const r = handle_summary_html(loaded);

  // Every <ppf:slot> and <ppf:template> should have been consumed by the renderer.
  EXPECT_EQ(r.body.find("<ppf:"), std::string::npos);
  EXPECT_EQ(r.body.find("</ppf:"), std::string::npos);
}

TEST_F(ServeHandlersClassificationTest, SummaryHtmlEscapesSpecialCharacters) {
  auto loaded             = load_from_file(model_->path());
  loaded.feature_names[0] = "<evil>&\"name\"";

  Response const r = handle_summary_html(loaded);

  EXPECT_EQ(r.body.find("<evil>"), std::string::npos);
  EXPECT_NE(r.body.find("&lt;evil&gt;"), std::string::npos);
  EXPECT_NE(r.body.find("&amp;"), std::string::npos);
  EXPECT_NE(r.body.find("&quot;"), std::string::npos);
}

TEST_F(ServeHandlersClassificationTest, PredictHappyPath) {
  auto const loaded = load_from_file(model_->path());

  std::string const csv = "Sepal.Length,Sepal.Width,Petal.Length,Petal.Width\n"
                          "5.1,3.5,1.4,0.2\n"
                          "6.7,3.0,5.2,2.3\n";

  Response const r = handle_predict(loaded, store, csv);

  ASSERT_EQ(r.status, 200);
  auto body = json::parse(r.body);
  ASSERT_TRUE(body.contains("predictions"));
  ASSERT_TRUE(body.at("predictions").is_array());
  EXPECT_EQ(body.at("predictions").size(), 2u);

  ASSERT_TRUE(body.contains("proportions"));
  ASSERT_TRUE(body.at("proportions").is_array());
  EXPECT_EQ(body.at("proportions").size(), 2u);
  EXPECT_EQ(body.at("proportions")[0].size(), loaded.group_names.size());
}

TEST_F(ServeHandlersClassificationTest, PredictReordersColumnsToModelOrder) {
  auto const loaded = load_from_file(model_->path());

  std::string const ordered  = "Sepal.Length,Sepal.Width,Petal.Length,Petal.Width\n"
                               "5.1,3.5,1.4,0.2\n";
  std::string const shuffled = "Petal.Width,Sepal.Length,Petal.Length,Sepal.Width\n"
                               "0.2,5.1,1.4,3.5\n";

  auto const a = json::parse(handle_predict(loaded, store, ordered).body);
  auto const b = json::parse(handle_predict(loaded, store, shuffled).body);

  // Predictions and proportions should match; `id`/`url` differ per store
  // entry, so compare just the data fields.
  EXPECT_EQ(a.at("predictions"), b.at("predictions"));
  EXPECT_EQ(a.at("proportions"), b.at("proportions"));
}

TEST_F(ServeHandlersClassificationTest, PredictRejectsMissingFeature) {
  auto const loaded = load_from_file(model_->path());

  std::string const csv = "Sepal.Length,Sepal.Width,Petal.Length\n"
                          "5.1,3.5,1.4\n";

  Response const r = handle_predict(loaded, store, csv);
  EXPECT_EQ(r.status, 400);
  EXPECT_NE(r.body.find("Petal.Width"), std::string::npos);
}

TEST_F(ServeHandlersClassificationTest, PredictIgnoresExtraColumns) {
  auto const loaded = load_from_file(model_->path());

  // CSVs carried over from training carry the response column ("Type" in
  // iris.csv). It must be silently ignored — only missing features are an
  // error.
  std::string const ordered    = "Sepal.Length,Sepal.Width,Petal.Length,Petal.Width\n"
                                 "5.1,3.5,1.4,0.2\n";
  std::string const with_extra = "Sepal.Length,Sepal.Width,Petal.Length,Petal.Width,Type\n"
                                 "5.1,3.5,1.4,0.2,setosa\n";

  Response const a = handle_predict(loaded, store, ordered);
  Response const b = handle_predict(loaded, store, with_extra);

  ASSERT_EQ(a.status, 200);
  ASSERT_EQ(b.status, 200);
  auto ja = json::parse(a.body);
  auto jb = json::parse(b.body);
  // Predictions and proportions are the same; the extras-as-labels path
  // adds a `metrics` field only when a response column is present.
  EXPECT_EQ(ja.at("predictions"), jb.at("predictions"));
  EXPECT_EQ(ja.at("proportions"), jb.at("proportions"));
  EXPECT_FALSE(ja.contains("metrics"));
  EXPECT_TRUE(jb.contains("metrics"));
}

TEST_F(ServeHandlersClassificationTest, PredictRejectsEmptyBody) {
  auto const loaded = load_from_file(model_->path());

  Response const r = handle_predict(loaded, store, "");
  EXPECT_EQ(r.status, 400);
}

TEST_F(ServeHandlersClassificationTest, PredictRejectsHeaderOnly) {
  auto const loaded = load_from_file(model_->path());

  Response const r = handle_predict(loaded, store, "Sepal.Length,Sepal.Width,Petal.Length,Petal.Width\n");
  EXPECT_EQ(r.status, 400);
}

TEST(ServeHandlersRegression, SummaryHtmlShowsRegressionMetrics) {
  TempFile model_file;
  model_file.clear();
  auto train = run_ppforest2("-q train -d " + MTCARS_CSV + " -n 5 -r 0 -s " + model_file.path());
  ASSERT_EQ(train.exit_code, 0);

  auto const loaded = load_from_file(model_file.path());

  Response const r = handle_summary_html(loaded);

  EXPECT_EQ(r.status, 200);
  EXPECT_NE(r.body.find("Regression"), std::string::npos);
  EXPECT_NE(r.body.find("MAE"), std::string::npos);
  EXPECT_NE(r.body.find("MSE"), std::string::npos);
  EXPECT_NE(r.body.find("R²"), std::string::npos);
  EXPECT_EQ(r.body.find("Accuracy"), std::string::npos);
}

TEST(ServeHandlersRegression, PredictOmitsProportions) {
  TempFile model_file;
  model_file.clear();
  auto train = run_ppforest2("-q train -d " + MTCARS_CSV + " -n 5 -r 0 -s " + model_file.path());
  ASSERT_EQ(train.exit_code, 0);

  auto const loaded = load_from_file(model_file.path());
  PredictionStore store;

  std::string const csv = "cyl,disp,hp,drat,wt,qsec,vs,am,gear,carb\n"
                          "6,160,110,3.9,2.62,16.46,0,1,4,4\n"
                          "8,360,245,3.21,3.57,15.84,0,0,3,4\n";

  Response const r = handle_predict(loaded, store, csv);

  ASSERT_EQ(r.status, 200);
  auto body = json::parse(r.body);
  ASSERT_TRUE(body.contains("predictions"));
  EXPECT_EQ(body.at("predictions").size(), 2u);
  EXPECT_FALSE(body.contains("proportions"));
}
