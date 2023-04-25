find_package(pybind11 REQUIRED)

pybind11_add_module(_${MODULE_NAME} MODULE pybind_main.cpp)
target_link_libraries(_${MODULE_NAME} PRIVATE ${MODULE_NAME} BaseGraph::core SamplableSet)

install(TARGETS _${MODULE_NAME} DESTINATION ./graphinf/${MODULE_NAME})
install(FILES ${MODULE_NAME}/__init__.py DESTINATION ./graphinf/${MODULE_NAME})
