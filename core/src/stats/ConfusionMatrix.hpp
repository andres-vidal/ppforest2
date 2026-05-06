/**
 * @file ConfusionMatrix.hpp
 * @brief Confusion matrix for classification model evaluation.
 */
#pragma once

#include "utils/Types.hpp"

#include <map>

namespace ppforest2::stats {
  /**
   * @brief Build a sorted mapping from unique group labels to contiguous indices.
   *
   * Includes labels from both `predictions` and `actual` so the confusion
   * matrix can hold cells for any predicted class — even classes the model
   * confused into that didn't appear in the ground truth (and vice versa).
   *
   * @return A map from label value to its 0-based index.
   */
  std::map<int, int> get_labels_map(types::GroupIdVector const& y_pred, types::GroupIdVector const& y);

  /**
   * @brief A confusion matrix comparing predicted vs actual group labels.
   *
   * Rows correspond to actual groups and columns to predicted groups.
   * Provides overall error rate, per-group error rates, JSON serialization,
   * and formatted terminal printing with diagonal highlighting and marginal errors.
   *
   * @code
   *   types::OutcomeVector preds = model.predict(x_test);
   *   ConfusionMatrix cm(preds, y_test);
   *
   *   float err = cm.error();                  // overall error rate
   *   auto per_class = cm.group_errors();      // per-group error rates
   * @endcode
   */
  struct ConfusionMatrix {
    types::Matrix<int> values;      ///< The NxN confusion matrix (actual x predicted).
    std::map<int, int> label_index; ///< Map from group label to matrix row/column index.

    /** @brief Default-construct an empty confusion matrix. */
    ConfusionMatrix() = default;

    /**
     * @brief Construct a confusion matrix from predictions and actual labels.
     * @param predictions The predicted group labels.
     * @param actual      The true group labels (must have the same size).
     * @throws std::invalid_argument If predictions and actual have different sizes.
     */
    ConfusionMatrix(types::GroupIdVector const& y_pred, types::GroupIdVector const& y);

    /**
     * @brief Compute per-group error rates.
     * @return A vector of error rates (one per group), where 0 = perfect, 1 = all wrong.
     */
    types::Vector<float> group_errors() const;

    /**
     * @brief Compute the overall error rate (1 - accuracy).
     * @return The proportion of misclassified samples.
     */
    float error() const;
  };
}
