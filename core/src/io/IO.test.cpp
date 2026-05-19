/**
 * @file IO.test.cpp
 * @brief Unit tests for CSV reading and file helper utilities.
 */
#include <gtest/gtest.h>

#include "io/IO.hpp"
#include "io/TempFile.hpp"
#include "utils/UserError.hpp"

#include <fstream>

using namespace ppforest2;

#ifndef PPFOREST2_DATA_DIR
#error "PPFOREST2_DATA_DIR must be defined"
#endif

static const std::string DATA_DIR  = PPFOREST2_DATA_DIR;
static std::string const IRIS_PATH = DATA_DIR + "/classification/iris.csv";

namespace {
  void write_csv(std::string const& path, std::string const& content) {
    std::ofstream out(path);
    out << content;
    out.close();
  }
}

TEST(CSVReadTest, AllNumericFeatures) {
  io::TempFile tmp(".csv");
  write_csv(tmp.path(), "a,b,label\n1.0,2.0,x\n3.0,4.0,y\n5.0,6.0,x\n");

  auto data = io::csv::read(tmp.path());

  EXPECT_EQ(data.x.rows(), 3);
  EXPECT_EQ(data.x.cols(), 2);
  EXPECT_FLOAT_EQ(data.x(0, 0), 1.0F);
  EXPECT_FLOAT_EQ(data.x(1, 1), 4.0F);
  EXPECT_EQ(data.y(0), 0);
  EXPECT_EQ(data.y(1), 1);
  EXPECT_EQ(data.y(2), 0);
}

TEST(CSVReadTest, CategoricalFeatureColumn) {
  io::TempFile tmp(".csv");
  write_csv(tmp.path(), "color,size,label\nred,1.0,A\nblue,2.0,B\nred,3.0,A\ngreen,4.0,B\n");

  auto data = io::csv::read(tmp.path());

  EXPECT_EQ(data.x.rows(), 4);
  EXPECT_EQ(data.x.cols(), 2);

  // "color" column: red=0, blue=1, green=2 (order of first appearance)
  EXPECT_FLOAT_EQ(data.x(0, 0), 0.0F); // red
  EXPECT_FLOAT_EQ(data.x(1, 0), 1.0F); // blue
  EXPECT_FLOAT_EQ(data.x(2, 0), 0.0F); // red
  EXPECT_FLOAT_EQ(data.x(3, 0), 2.0F); // green

  // "size" column: numeric, unchanged
  EXPECT_FLOAT_EQ(data.x(0, 1), 1.0F);
  EXPECT_FLOAT_EQ(data.x(1, 1), 2.0F);
}

TEST(CSVReadTest, MultipleCategoricalColumns) {
  io::TempFile tmp(".csv");
  write_csv(tmp.path(), "color,shape,val,label\nred,circle,1.0,X\nblue,square,2.0,Y\nred,circle,3.0,X\n");

  auto data = io::csv::read(tmp.path());

  EXPECT_EQ(data.x.rows(), 3);
  EXPECT_EQ(data.x.cols(), 3);

  // color: red=0, blue=1
  EXPECT_FLOAT_EQ(data.x(0, 0), 0.0F);
  EXPECT_FLOAT_EQ(data.x(1, 0), 1.0F);
  EXPECT_FLOAT_EQ(data.x(2, 0), 0.0F);

  // shape: circle=0, square=1
  EXPECT_FLOAT_EQ(data.x(0, 1), 0.0F);
  EXPECT_FLOAT_EQ(data.x(1, 1), 1.0F);
  EXPECT_FLOAT_EQ(data.x(2, 1), 0.0F);

  // val: numeric
  EXPECT_FLOAT_EQ(data.x(0, 2), 1.0F);
}

TEST(CSVReadTest, GroupNamesPopulated) {
  io::TempFile tmp(".csv");
  write_csv(tmp.path(), "x,label\n1.0,setosa\n2.0,virginica\n3.0,setosa\n");

  auto data = io::csv::read(tmp.path());

  ASSERT_EQ(data.group_names.size(), 2U);
  EXPECT_EQ(data.group_names[0], "setosa");
  EXPECT_EQ(data.group_names[1], "virginica");
}

TEST(CSVReadTest, GroupNamesPreservedAfterSort) {
  io::TempFile tmp(".csv");
  write_csv(tmp.path(), "x,label\n1.0,B\n2.0,A\n3.0,B\n4.0,A\n");

  auto data = io::csv::read_sorted(tmp.path());

  ASSERT_EQ(data.group_names.size(), 2U);
  EXPECT_EQ(data.group_names[0], "B");
  EXPECT_EQ(data.group_names[1], "A");
}

// Mode auto-detection: regression iff any y value has fractional or
// scientific notation; integer-coded labels remain classification.

TEST(CSVReadTest, RegressionDetectedFromFractionalY) {
  io::TempFile tmp(".csv");
  write_csv(tmp.path(), "a,b,y\n1.0,2.0,0.5\n3.0,4.0,1.7\n5.0,6.0,2.3\n");

  auto data = io::csv::read_sorted(tmp.path());

  EXPECT_TRUE(data.group_names.empty()) << "Fractional y should auto-detect as regression";
  // Sorted ascending by y → 0.5, 1.7, 2.3.
  EXPECT_FLOAT_EQ(data.y(0), 0.5F);
  EXPECT_FLOAT_EQ(data.y(1), 1.7F);
  EXPECT_FLOAT_EQ(data.y(2), 2.3F);
}

TEST(CSVReadTest, IntegerCodedLabelsRemainClassification) {
  io::TempFile tmp(".csv");
  write_csv(tmp.path(), "a,b,label\n1.0,2.0,1\n3.0,4.0,2\n5.0,6.0,3\n7.0,8.0,1\n");

  auto data = io::csv::read_sorted(tmp.path());

  ASSERT_EQ(data.group_names.size(), 3U) << "Integer-coded labels should map to classification group_names";
  EXPECT_EQ(data.group_names[0], "1");
  EXPECT_EQ(data.group_names[1], "2");
  EXPECT_EQ(data.group_names[2], "3");
}

TEST(CSVReadTest, ScientificNotationDetectsRegression) {
  io::TempFile tmp(".csv");
  write_csv(tmp.path(), "a,b,y\n1.0,2.0,1e2\n3.0,4.0,2e2\n");

  auto data = io::csv::read_sorted(tmp.path());

  EXPECT_TRUE(data.group_names.empty()) << "y in scientific notation should auto-detect as regression";
  EXPECT_FLOAT_EQ(data.y(0), 100.0F);
  EXPECT_FLOAT_EQ(data.y(1), 200.0F);
}

TEST(CSVReadTest, NonNumericLabelStaysClassification) {
  // Already covered by AllNumericFeatures and GroupNamesPopulated, but
  // assert it explicitly: non-numeric y is unambiguously classification.
  io::TempFile tmp(".csv");
  write_csv(tmp.path(), "a,b,label\n1.0,2.0,foo\n3.0,4.0,bar\n");

  auto data = io::csv::read_sorted(tmp.path());

  ASSERT_EQ(data.group_names.size(), 2U);
}

TEST(CSVReadTest, CrabsDatasetWithCategoricalSex) {
  auto data = io::csv::read(PPFOREST2_DATA_DIR "/classification/crabs.csv");

  EXPECT_EQ(data.x.cols(), 7); // sex + index + 5 morphometrics
  EXPECT_EQ(data.x.rows(), 200);

  // "sex" is the first column and is categorical (M/F)
  // All values should be 0 or 1
  for (int i = 0; i < data.x.rows(); ++i) {
    types::Feature val = data.x(i, 0);
    EXPECT_TRUE(val == 0.0F || val == 1.0F) << "Row " << i << " sex=" << val;
  }
}

// ---------------------------------------------------------------------------
// File helpers — io::json::ensure_extension, check_*_not_exists
// ---------------------------------------------------------------------------

/**
 * @brief Death-test predicate: matches any non-zero exit code.
 */
class ExitedWithNonZero {
public:
  bool operator()(int exit_status) const {
    // clang-format off
    #ifdef _WIN32
    return exit_status != 0;
    #else
    return testing::ExitedWithCode(0)(exit_status) && WIFEXITED(exit_status);
    #endif
    // clang-format on
  }
};

/* Path already ending in .json is returned unchanged. */
TEST(FileHelpers, EnsureJsonExtensionWithExtension) {
  EXPECT_EQ(io::json::ensure_extension("model.json"), "model.json");
}

/* Path without extension gets .json appended. */
TEST(FileHelpers, EnsureJsonExtensionWithoutExtension) {
  EXPECT_EQ(io::json::ensure_extension("model"), "model.json");
}

/* Non-.json extension gets .json added (e.g. .txt -> .txt.json). */
TEST(FileHelpers, EnsureJsonExtensionWithOtherExtension) {
  EXPECT_EQ(io::json::ensure_extension("model.txt"), "model.txt.json");
}

/* Full path without extension gets .json appended. */
TEST(FileHelpers, EnsureJsonExtensionWithPath) {
  EXPECT_EQ(io::json::ensure_extension("/tmp/model"), "/tmp/model.json");
}

/* check_file_not_exists succeeds for a nonexistent path. */
TEST(FileHelpers, CheckFileNotExistsOnNonexistent) {
  io::check_file_not_exists("/nonexistent/path/that/doesnt/exist.json");
}

/* check_file_not_exists throws for an existing file. */
TEST(FileHelpers, CheckFileNotExistsOnExisting) {
  EXPECT_THROW(io::check_file_not_exists(IRIS_PATH), ppforest2::UserError);
}

/* check_dir_not_exists succeeds for a nonexistent path. */
TEST(FileHelpers, CheckDirNotExistsOnNonexistent) {
  io::check_dir_not_exists("/nonexistent/path/that/doesnt/exist");
}

/* check_dir_not_exists throws for an existing directory. */
TEST(FileHelpers, CheckDirNotExistsOnExisting) {
  EXPECT_THROW(io::check_dir_not_exists(DATA_DIR), ppforest2::UserError);
}

/* check_file_exists succeeds for an existing file. */
TEST(FileHelpers, CheckFileExistsOnExisting) {
  io::check_file_exists(IRIS_PATH);
}

/* check_file_exists throws for a nonexistent file. */
TEST(FileHelpers, CheckFileExistsOnNonexistent) {
  EXPECT_THROW(io::check_file_exists("/nonexistent/path.csv"), ppforest2::UserError);
}

/* read_features_from_string parses a feature-only CSV body. */
TEST(ReadFeaturesFromString, ParsesNumericCsv) {
  auto set = io::csv::read_features_from_string("a,b,c\n1.0,2.0,3.0\n4.0,5.0,6.0\n");

  ASSERT_EQ(set.feature_names.size(), 3U);
  EXPECT_EQ(set.feature_names[0], "a");
  EXPECT_EQ(set.feature_names[2], "c");

  ASSERT_EQ(set.x.rows(), 2);
  ASSERT_EQ(set.x.cols(), 3);
  EXPECT_FLOAT_EQ(set.x(0, 0), 1.0F);
  EXPECT_FLOAT_EQ(set.x(1, 2), 6.0F);
}

TEST(ReadFeaturesFromString, EmptyBodyThrows) {
  EXPECT_THROW(io::csv::read_features_from_string(""), ppforest2::UserError);
}

TEST(ReadFeaturesFromString, HeaderOnlyThrows) {
  EXPECT_THROW(io::csv::read_features_from_string("a,b,c\n"), ppforest2::UserError);
}

TEST(ReadFeaturesFromString, RaggedRowsThrow) {
  EXPECT_THROW(io::csv::read_features_from_string("a,b,c\n1,2,3\n4,5\n"), ppforest2::UserError);
}
