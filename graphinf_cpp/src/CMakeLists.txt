file(GLOB_RECURSE GRAPHINF_SRC "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
add_library(graphinf ${GRAPHINF_SRC})


target_link_libraries(graphinf ${BASEGRAPH} ${SAMPLABLESET})
set_target_properties(graphinf PROPERTIES
    LINKER_LANGUAGE CXX
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
