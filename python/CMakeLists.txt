find_package(pybind11 REQUIRED)

pybind11_add_module(_core MODULE pybind_main.cpp)
target_link_libraries(_core PRIVATE graphinf BaseGraph::core ${SAMPLABLESET})

install(TARGETS _core DESTINATION ../graphinf)
# install(FILES graphinf/__init__.py DESTINATION ./graphinf/${MODULE_NAME})
