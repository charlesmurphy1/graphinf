cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(graphinf LANGUAGES CXX VERSION 0.3.0
        DESCRIPTION "A library for Markov chain Monte-Carlo on graphs.")


option(DEBUG_MODE "check consistency of objects at runtime" OFF)
option(BUILD_TESTS "build gtest unit tests" OFF)

set(CMAKE_CXX_STANDARD 11)
set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
if (APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++${CMAKE_CXX_STANDARD} -stdlib=libc++")
endif()

find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
  option(GIT_SUBMODULE "Check submodules during build" ON)
  if(GIT_SUBMODULE)
    message(STATUS "Submodule update")
    execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    RESULT_VARIABLE GIT_SUBMOD_RESULT)
    if(NOT GIT_SUBMOD_RESULT EQUAL "0")
      message(FATAL_ERROR "git submodule --init failed with ${GIT_SUBMOD_RESULT}, please check whether the submodules have been properly installed.")
    endif()
  endif()
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/ext/base_graph/include)
include_directories(${PROJECT_SOURCE_DIR}/ext/SamplableSet/src)

add_subdirectory(${PROJECT_SOURCE_DIR}/ext/SamplableSet/src SamplableSet)
add_subdirectory(src)

if (SKBUILD)
    set(CMAKE_BUILD_TYPE Release)
    add_subdirectory(python)
endif()
if (BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
