cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(GraphInf)
set(CXX_STANDARD 14)
set(CXX_STANDARD_REQUIRED on)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

option(DEBUG_MODE "check consistency of objects at runtime" OFF)
option(BUILD_TESTS "build gtest unit tests" OFF)

if (DEBUG_MODE)
    add_compile_definitions(DEBUG)
endif()

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

    find_package(pybind11 REQUIRED)
endif()


include_directories(include)
include_directories(base_graph/include)
include_directories(SamplableSet/src)
set(BASEGRAPH "${CMAKE_CURRENT_SOURCE_DIR}/base_graph/build/libBaseGraph.a")
set(SAMPLABLESET "${CMAKE_CURRENT_SOURCE_DIR}/SamplableSet/src/build/libsamplableset.a")

add_subdirectory(src)

if(SKBUILD)
    pybind11_add_module(_graphinf ${CMAKE_SOURCE_DIR}/pybind_wrapper/pybind_main.cpp)
    set_target_properties(_graphinf PROPERTIES CXX_STANDARD 14)

    set_target_properties(graphinf PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
    target_link_libraries(_graphinf PRIVATE ${BASEGRAPH} ${SAMPLABLESET} graphinf)
    install(TARGETS _graphinf DESTINATION _graphinf)
endif()

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()