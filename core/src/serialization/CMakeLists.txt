add_library(ppforest2-serialization OBJECT
  ExportValidation.cpp
  Json.cpp)

target_link_libraries(ppforest2-serialization PRIVATE ppforest2-compile nlohmann_json::nlohmann_json)
