#pragma once

#include "models/TreeNode.hpp"
#include "models/TreeBranch.hpp"
#include "models/TreeLeaf.hpp"
#include "models/Tree.hpp"
#include "models/Forest.hpp"
#include "models/Evaluation.hpp"
#include "stats/ConfusionMatrix.hpp"
#include "stats/Metrics.hpp"

#include <nlohmann/json.hpp>
#include <optional>
#include <ostream>

/**
 * @brief JSON serialization and deserialization for ppforest2 models.
 *
 * Uses nlohmann/json.  Provides to_json() for serializing trees,
 * forests, confusion matrices, and variable importance to JSON.
 * Deserialization uses `j.get<T>()` via nlohmann ADL:
 *
 * @code
 *   // Serialize and write to file:
 *   auto j = serialization::to_json(forest);
 *   io::json::write_file(j, "model.json");
 *
 *   // Read a full export (model + config + meta):
 *   auto j2 = io::json::read_file("model.json");
 *   auto e  = j2.get<Export<Forest>>();  // e.model, e.groups, e.spec
 *
 *   // Read bare model block (integer labels):
 *   Tree tree = model_json.get<Tree>();
 * @endcode
 */
namespace ppforest2::serialization {
  using json = nlohmann::json;

  /** @brief Re-export of `types::Names` for serialization callers. */
  using Names = types::Names;

  /** @brief Re-export of `stats::Metrics` for serialization callers. */
  using Metrics = stats::Metrics;

  /**
   * @brief A model bundled with its export metadata and optional metrics.
   *
   * Represents the full JSON export format:
   * `{ model_type, model, config, meta, [metrics] }`.
   *
   * Use `j.get<Export<Tree>>()` or `j.get<Export<Model::Ptr>>()` to
   * deserialize a full export, and `model_export.to_json()` to serialize.
   *
   * For `Export<Model::Ptr>`, construct with training data to compute
   * metrics automatically, or use `compute_metrics()` after construction.
   */
  template<typename T> struct Export {
    T model;
    /** @brief Group name vector — empty for regression (no group concept), non-empty for classification. */
    Names groups;
    TrainingSpec::Ptr spec;
    int n_observations = 0;
    int n_features     = 0;
    types::Names feature_names;

    /** @name Optional metrics — serialized by to_json() when present. */
    ///@{
    std::optional<VariableImportance> variable_importance = std::nullopt;
    std::optional<Metrics> training_metrics               = std::nullopt;
    std::optional<Metrics> oob_metrics                    = std::nullopt;
    ///@}

    /** @brief Serialize to JSON. Only defined for Export<Model::Ptr>. */
    json to_json() const;

    /**
     * @brief Compute and store metrics on this export.
     *
     * Only defined for Export<Model::Ptr>.
     */
    void compute_metrics(types::FeatureMatrix const& x, types::OutcomeVector const& y);
  };

  // Explicit specialization declarations for Export<Model::Ptr>.
  template<> json Export<Model::Ptr>::to_json() const;
  template<> void Export<Model::Ptr>::compute_metrics(types::FeatureMatrix const&, types::OutcomeVector const&);

  /** @brief Visitor that serializes a tree node to JSON. */
  class JsonNodeVisitor : public TreeNode::Visitor {
  public:
    json result;
    /** @brief When non-empty, leaf values and group sets are written as label strings. */
    Names const group_names;

    JsonNodeVisitor() = default;
    explicit JsonNodeVisitor(Names group_names)
        : group_names(std::move(group_names)) {}

    void visit(TreeBranch const& node) override;
    void visit(TreeLeaf const& node) override;
  };

  /** @brief Visitor that serializes a model (Tree or Forest) to JSON. */
  class JsonModelVisitor : public Model::Visitor {
  public:
    json result;
    /** @brief When non-empty, the model's leaves and groups are written as label strings. */
    Names const group_names;

    JsonModelVisitor() = default;
    explicit JsonModelVisitor(Names group_names)
        : group_names(std::move(group_names)) {}

    void visit(Tree const& tree) override;
    void visit(Forest const& forest) override;
  };

  /**
   * @brief Serialize a prediction vector as JSON.
   *
   * With non-empty @p names, the predictions are interpreted as integer-coded
   * class labels and mapped to label strings (`names[code]`). With empty
   * @p names, the values are passed through verbatim as a flat JSON array —
   * letting callers stay mode-agnostic (regression: continuous floats;
   * classification without label metadata: integer codes).
   */
  json to_json(types::OutcomeVector const& y, types::Names const& names);

  /** @name Serialization */
  ///@{
  json to_json(Model const& model);
  json to_json(TreeNode const& node);
  json to_json(Tree const& tree);
  json to_json(BaggedTree const& tree);
  json to_json(Forest const& forest);
  json to_json(stats::ConfusionMatrix const& cm);
  json to_json(stats::ClassificationMetrics const& cm);
  json to_json(VariableImportance const& vi);
  json to_json(stats::RegressionMetrics const& rm);
  json to_json(Metrics const& metrics);
  json to_json(types::FeatureMatrix const& matrix);
  ///@}

  /** @name Labeled serialization (uses group names instead of integer codes) */
  ///@{
  json to_json(Model const& model, Names const& group_names);
  json to_json(TreeNode const& node, Names const& group_names);
  json to_json(Tree const& tree, Names const& group_names);
  json to_json(BaggedTree const& tree, Names const& group_names);
  json to_json(Forest const& forest, Names const& group_names);
  json to_json(stats::ConfusionMatrix const& cm, Names const& group_names);
  json to_json(stats::ClassificationMetrics const& cm, Names const& group_names);
  json to_json(stats::RegressionMetrics const& rm, Names const& group_names);
  json to_json(Metrics const& metrics, Names const& group_names);
  ///@}

  /** @name Optional serialization — `nullopt` round-trips as JSON `null`. */
  ///@{
  json to_json(std::optional<VariableImportance> const& vi);
  json to_json(std::optional<Metrics> const& metrics);
  json to_json(std::optional<Metrics> const& metrics, Names const& group_names);
  ///@}

  /** @name Deserialization */
  ///@{

  /**
   * @brief Deserialize a value block (confusion matrix, VI, metrics, …).
   *
   * Models go through `j.get<Export<Model::Ptr>>()` (or the typed
   * `Export<Tree::Ptr>` / `Export<Forest::Ptr>` overloads) which rehydrate
   * the real `TrainingSpec` from the JSON's `config` block. There's no
   * standalone `from_json<Tree::Ptr>` because constructing a concrete tree
   * without its spec would require a placeholder, which masks the missing
   * config silently.
   */
  template<typename T> T from_json(json const& j);

  template<> stats::ConfusionMatrix from_json<stats::ConfusionMatrix>(json const& j);
  template<> stats::ClassificationMetrics from_json<stats::ClassificationMetrics>(json const& j);
  template<> VariableImportance from_json<VariableImportance>(json const& j);
  template<> stats::RegressionMetrics from_json<stats::RegressionMetrics>(json const& j);

  /** @brief Deserialize a `Metrics` block; mode picks the variant alternative. */
  Metrics metrics_from_json(json const& j, types::Mode mode);
  ///@}

  /** @name Stream operators */
  ///@{
  std::ostream& operator<<(std::ostream& os, TreeNode const& node);
  std::ostream& operator<<(std::ostream& os, TreeBranch const& condition);
  std::ostream& operator<<(std::ostream& os, TreeLeaf const& response);
  std::ostream& operator<<(std::ostream& os, Tree const& tree);
  std::ostream& operator<<(std::ostream& os, Forest const& forest);

  template<typename V> std::ostream& operator<<(std::ostream& ostream, std::vector<V> const& vec) {
    json json_vector(vec);
    return ostream << json_vector.dump();
  }

  template<typename V, typename C1, typename C2>
  std::ostream& operator<<(std::ostream& ostream, std::set<V, C1, C2> const& set) {
    json json_set(set);
    return ostream << json_set.dump();
  }

  template<typename K, typename V> std::ostream& operator<<(std::ostream& ostream, std::map<K, V> const& map) {
    json json_map(map);
    return ostream << json_map.dump();
  }

  template<typename V> std::ostream& operator<<(std::ostream& ostream, std::map<int, V> const& map) {
    std::map<std::string, V> string_map;

    for (auto const& [key, val] : map) {
      string_map[std::to_string(key)] = val;
    }

    json json_map(string_map);
    return ostream << json_map.dump();
  }

  ///@}
}

// ADL serializer specializations — enables j.get<T>() for all types.
namespace nlohmann {
  template<> struct adl_serializer<ppforest2::stats::ConfusionMatrix> {
    static ppforest2::stats::ConfusionMatrix from_json(json const& j) {
      return ppforest2::serialization::from_json<ppforest2::stats::ConfusionMatrix>(j);
    }
  };

  template<> struct adl_serializer<ppforest2::stats::ClassificationMetrics> {
    static ppforest2::stats::ClassificationMetrics from_json(json const& j) {
      return ppforest2::serialization::from_json<ppforest2::stats::ClassificationMetrics>(j);
    }
  };

  template<> struct adl_serializer<ppforest2::VariableImportance> {
    static ppforest2::VariableImportance from_json(json const& j) {
      return ppforest2::serialization::from_json<ppforest2::VariableImportance>(j);
    }
  };

  template<> struct adl_serializer<ppforest2::stats::RegressionMetrics> {
    static ppforest2::stats::RegressionMetrics from_json(json const& j) {
      return ppforest2::serialization::from_json<ppforest2::stats::RegressionMetrics>(j);
    }
  };

  template<> struct adl_serializer<ppforest2::serialization::Export<ppforest2::Tree::Ptr>> {
    static ppforest2::serialization::Export<ppforest2::Tree::Ptr> from_json(json const& j);
  };

  template<> struct adl_serializer<ppforest2::serialization::Export<ppforest2::Forest::Ptr>> {
    static ppforest2::serialization::Export<ppforest2::Forest::Ptr> from_json(json const& j);
  };

  template<> struct adl_serializer<ppforest2::serialization::Export<ppforest2::Model::Ptr>> {
    static ppforest2::serialization::Export<ppforest2::Model::Ptr> from_json(json const& j);
  };

  /**
   * @brief Eigen::Matrix ↔ JSON.
   *
   * Column vectors (`Cols == 1`) round-trip as a flat array;
   * general matrices round-trip as a nested array of rows.
   *
   * Lets `j[key] = matrix` and `j[key].get<Matrix>()` work directly
   * for any Eigen dynamic-size matrix or vector.
   */
  template<typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
  struct adl_serializer<Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>> {
    using Matrix = Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>;

    static void to_json(json& j, Matrix const& m) {
      if constexpr (Cols == 1) {
        j = json::array();
        for (Eigen::Index i = 0; i < m.size(); ++i) {
          j.push_back(m(i));
        }
      } else {
        j = json::array();
        for (Eigen::Index i = 0; i < m.rows(); ++i) {
          json row = json::array();
          for (Eigen::Index k = 0; k < m.cols(); ++k) {
            row.push_back(m(i, k));
          }
          j.push_back(std::move(row));
        }
      }
    }

    static void from_json(json const& j, Matrix& m) {
      if constexpr (Cols == 1) {
        m.resize(static_cast<Eigen::Index>(j.size()));
        for (std::size_t i = 0; i < j.size(); ++i) {
          m(static_cast<Eigen::Index>(i)) = j[i].template get<Scalar>();
        }
      } else {
        Eigen::Index const rows = static_cast<Eigen::Index>(j.size());
        Eigen::Index const cols = rows > 0 ? static_cast<Eigen::Index>(j.front().size()) : 0;
        m.resize(rows, cols);
        for (Eigen::Index i = 0; i < rows; ++i) {
          auto const& row = j[static_cast<std::size_t>(i)];
          for (Eigen::Index k = 0; k < cols; ++k) {
            m(i, k) = row[static_cast<std::size_t>(k)].template get<Scalar>();
          }
        }
      }
    }
  };
}
