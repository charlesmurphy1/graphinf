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

# if (NOT SKBUILD)
#     include(CMakePackageConfigHelpers)
#     include(GNUInstallDirs)

#     # Install headers
#     install(
#         DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/GraphInf"
#         DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
#     )

#     # Install targets
#     install(
#         TARGETS ${EXTENSION_NAME}
#         EXPORT ${PROJECT_NAME}-targets
#         LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#         ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
#         RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
#         INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
#     )

#     # Install export target and config for find_package
#     set(BASEGRAPH_EXPORT_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")
#     install(
#         EXPORT ${PROJECT_NAME}-targets
#         NAMESPACE BaseGraph::
#         DESTINATION ${BASEGRAPH_EXPORT_DIR}
#     )

#     configure_package_config_file(
#         "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
#         "${CMAKE_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
#         INSTALL_DESTINATION ${BASEGRAPH_EXPORT_DIR}
#     )
#     write_basic_package_version_file(
#         "${CMAKE_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
#         VERSION ${PROJECT_VERSION}
#         COMPATIBILITY SameMajorVersion
#     )
#     install(FILES
#         "${CMAKE_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
#         "${CMAKE_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
#         DESTINATION ${BASEGRAPH_EXPORT_DIR}
#     )
# endif()
