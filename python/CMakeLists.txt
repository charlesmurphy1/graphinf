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

find_package(pybind11 CONFIG REQUIRED)


file(GLOB_RECURSE CORE_BINDINGS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

pybind11_add_module(libgraphinf MODULE ${CORE_BINDINGS_SRC})
# target_link_libraries(libgraphinf PRIVATE core)
set_target_properties(libgraphinf PROPERTIES CXX_STANDARD 14)
target_compile_definitions(libgraphinf PRIVATE VERSION_INFO=${PROJECT_VERSION})

message(STATUS "SKBUILD:" ${SKBUILD})
if (SKBUILD)
    install(TARGETS libgraphinf DESTINATION ./graphinf)
else()
    set_target_properties(libgraphinf PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python-lib")
endif()
