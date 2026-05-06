# Non-strategy model code (Tree, Forest, TreeNode, TrainingSpec, …).
# Kept as an OBJECT library because this code is referenced directly on
# every training path — there's no self-registration here, so on-demand
# linkage via `$<TARGET_OBJECTS:…>` into `ppforest2-core` is sufficient.
#
# Concrete strategies (PDA, MinSize, Uniform, …) live under
# `strategies/` as a separate STATIC library — they self-register via
# `PPFOREST2_REGISTER_STRATEGY` and need WHOLE_ARCHIVE linkage or the
# initializers get dead-stripped. See `strategies/CMakeLists.txt`.
add_library(ppforest2-models OBJECT
  Model.cpp
  Tree.cpp
  ClassificationTree.cpp
  RegressionTree.cpp
  Forest.cpp
  ClassificationForest.cpp
  RegressionForest.cpp
  TreeNode.cpp
  TreeBranch.cpp
  TreeLeaf.cpp
  Evaluation.cpp
  TrainingSpec.cpp
  Projector.cpp
  Visualization.cpp)

target_link_libraries(ppforest2-models PRIVATE ppforest2-compile nlohmann_json::nlohmann_json)

add_subdirectory(strategies)
