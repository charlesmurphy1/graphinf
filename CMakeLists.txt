cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(graphinf LANGUAGES CXX VERSION 0.3.0
        DESCRIPTION "A library for Markov chain Monte-Carlo on graphs.")


option(DEBUG_MODE "check consistency of objects at runtime" OFF)
option(BUILD_TESTS "build gtest unit tests" OFF)

set(CMAKE_CXX_STANDARD 11)
set(CXX_STANDARD_REQUIRED ON)
if (APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++${CMAKE_CXX_STANDARD} -stdlib=libc++")
endif()

find_package(BaseGraph REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/ext/base_graph/include)
include_directories(${PROJECT_SOURCE_DIR}/ext/SamplableSet/src)
set(SAMPLABLESET "${CMAKE_CURRENT_SOURCE_DIR}/ext/SamplableSet/src/build/libsamplableset.a")

add_subdirectory(src)
if (SKBUILD)
    set(CMAKE_BUILD_TYPE Release)
    add_subdirectory(python)
endif()
if (BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
