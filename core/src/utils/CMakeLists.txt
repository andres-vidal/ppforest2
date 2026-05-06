add_library(ppforest2-utils OBJECT
  Invariant.cpp
  JsonReader.cpp
  UserError.cpp
  System.cpp)

target_link_libraries(ppforest2-utils PRIVATE ppforest2-compile nlohmann_json::nlohmann_json)
