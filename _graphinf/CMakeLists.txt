cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(GraphInf CXX)
set(CXX_STANDARD 11)
set(CXX_STANDARD_REQUIRED on)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

option(DEBUG_MODE "check consistency of objects at runtime" OFF)
option(BUILD_TESTS "build gtest unit tests" OFF)

if (DEBUG_MODE)
    add_compile_definitions(DEBUG)
endif()

include_directories(include)
include_directories(base_graph/include)
include_directories(SamplableSet/src)
set(BASEGRAPH "${CMAKE_CURRENT_SOURCE_DIR}/base_graph/build/libBaseGraph.a")
set(SAMPLABLESET "${CMAKE_CURRENT_SOURCE_DIR}/SamplableSet/src/build/libsamplableset.a")

add_subdirectory(src)


if (BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
