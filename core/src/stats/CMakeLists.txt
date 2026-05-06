add_library(ppforest2-stats OBJECT
  Stats.cpp
  Uniform.cpp
  Normal.cpp
  GroupPartition.cpp
  ConfusionMatrix.cpp
  Metrics.cpp
  Simulation.cpp)

target_link_libraries(ppforest2-stats PRIVATE ppforest2-compile)
