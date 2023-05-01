if(SKBUILD)
  # Scikit-Build does not add your site-packages to the search path
  # automatically, so we need to add it _or_ the pybind11 specific directory
  # here.
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c
            "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE _tmp_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
  list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")
endif()

find_package(pybind11 REQUIRED)

pybind11_add_module(_${PROJECT_NAME} MODULE pybind_main.cpp)
set_target_properties(_${PROJECT_NAME} PROPERTIES CXX_STANDARD 14)
target_link_libraries(_${PROJECT_NAME} 
                      PRIVATE graphinf 
                      "${PROJECT_BINARY_DIR}/SamplableSet/libsamplableset.a")
target_include_directories(_${PROJECT_NAME} PRIVATE "${PROJECT_SOURCE_DIR}/ext/base_graph/include")
install(TARGETS _${PROJECT_NAME} DESTINATION ./graphinf)
