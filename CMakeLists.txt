cmake_minimum_required(VERSION 3.18...3.22)

project(GraphInf VERSION 0.3.0 LANGUAGES CXX 
        DESCRIPTION "A library for Markov chain Monte-Carlo on graphs and graph inference.")

option(DEBUG_MODE "check consistency of objects at runtime" OFF)
option(BUILD_TESTS "build gtest unit tests" OFF)
option(BUILD_BINDINGS "build python bindings" OFF)

set(CXX_STANDARD 17)
set(CXX_STANDARD_REQUIRED on)
if (APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -stdlib=libc++")
endif()

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if (DEBUG_MODE)
    add_compile_definitions(DEBUG)
endif()

add_library(core INTERFACE)
target_include_directories(core INTERFACE
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/ext/base_graph/include)
include_directories(${PROJECT_SOURCE_DIR}/ext/SamplableSet/src)
set(SAMPLABLESET "${CMAKE_CURRENT_SOURCE_DIR}/ext/SamplableSet/src/build/libsamplableset.a")
add_subdirectory(src)

if (BUILD_BINDINGS OR SKBUILD)
    set(CMAKE_BUILD_TYPE Release)
    add_subdirectory(python)
endif()
if (BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Install GraphInf on system
if (NOT SKBUILD)
    include(CMakePackageConfigHelpers)
    include(GNUInstallDirs)

    # Install headers
    install(
        DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/GraphInf"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    # Install targets
    set(GRAPHINF_EXPORT_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")
    install(
        TARGETS core
        EXPORT ${PROJECT_NAME}-targets
    )

    # Install export target and config for find_package
    set(GRAPHINF_EXPORT_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")
    install(
        EXPORT ${PROJECT_NAME}-targets
        NAMESPACE ${PROJECT_NAME}::
        DESTINATION ${GRAPHINF_EXPORT_DIR}
    )

    # Install version information
    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
        "${CMAKE_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
        INSTALL_DESTINATION ${GRAPHINF_EXPORT_DIR}
    )
    write_basic_package_version_file(
        "${CMAKE_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )
    install(FILES
        "${CMAKE_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
        "${CMAKE_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
        DESTINATION ${GRAPHINF_EXPORT_DIR}
    )
endif()