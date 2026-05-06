# Concrete strategy implementations. STATIC (not OBJECT) on purpose:
# every concrete strategy self-registers at static-init time via
# `PPFOREST2_REGISTER_STRATEGY` — if the containing object file isn't
# directly referenced by consumer code, the linker dead-strips it and
# the registration never fires, so `*::from_json("name")` throws
# "Unknown strategy" at runtime.
#
# Consumers should WHOLE_ARCHIVE-link this target so every .o survives
# the dead-strip pass. See:
# - `core/CMakeLists.txt` for the three executables (WHOLE_ARCHIVE via
#   `$<LINK_LIBRARY:WHOLE_ARCHIVE,ppforest2-strategies>`).
# - `bindings/R/src/Makevars.in` for the R shared library
#   (platform-specific `-force_load` / `--whole-archive`).
#
# Keeping just these 14 .cpp files in the whole-archived target (rather
# than all of `ppforest2-models`) means non-strategy code like
# `Tree.cpp`, `Forest.cpp`, etc. stays on-demand in `ppforest2-core` —
# code paths a consumer doesn't use (e.g. the R bindings don't call
# `io::csv::read_*`) don't get pulled in just because one archive had
# to be whole-archived.
add_library(ppforest2-strategies STATIC
  binarize/Binarization.cpp
  binarize/Disabled.cpp
  binarize/LargestGap.cpp
  cutpoint/Cutpoint.cpp
  cutpoint/MeanOfMeans.cpp
  grouping/ByCutpoint.cpp
  grouping/ByLabel.cpp
  grouping/Grouping.cpp
  leaf/MajorityVote.cpp
  leaf/MeanResponse.cpp
  pp/PDA.cpp
  pp/ProjectionPursuit.cpp
  stop/CompositeStop.cpp
  stop/MaxDepth.cpp
  stop/MinSize.cpp
  stop/MinVariance.cpp
  stop/PureNode.cpp
  stop/StopRule.cpp
  vars/All.cpp
  vars/Uniform.cpp
  vars/VariableSelection.cpp)

target_link_libraries(ppforest2-strategies PRIVATE ppforest2-compile nlohmann_json::nlohmann_json)
