find_package(GTest)

if (NOT ${GTEST_FOUND} OR ${GTEST_FOUND})
    include(FetchContent)
    FetchContent_Declare(
            googletest
            # Specify the commit you depend on and update it regularly.
            URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif ()

file(GLOB_RECURSE GRAPH_TEST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/test_graph/test_*.cpp)
file(GLOB_RECURSE DATA_TEST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/test_data/test_*.cpp)
file(GLOB_RECURSE GENERATORS_TEST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/test_generators/test_*.cpp)
file(GLOB_RECURSE UTILITY_TEST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/test_utility/test_*.cpp)
file(GLOB_RECURSE PROPOSERS_TEST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/test_proposers/test_*.cpp)
file(GLOB_RECURSE PRIORS_TEST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/test_priors/test_*.cpp)

add_executable(test_graph ${GRAPH_TEST_SRC})
add_executable(test_data ${DATA_TEST_SRC})
add_executable(test_generators ${GENERATORS_TEST_SRC})
add_executable(test_utility ${UTILITY_TEST_SRC})
add_executable(test_proposers ${PROPOSERS_TEST_SRC})
add_executable(test_priors ${PRIORS_TEST_SRC})

set(allGTests
    test_graph
    test_data
    test_generators
    test_utility    
    test_proposers
    test_priors
    )


foreach(UnitTest ${allGTests})
    target_link_libraries(${UnitTest} gtest gtest_main graphinf ${PROJECT_BINARY_DIR}/SamplableSet/libsamplableset.a)
    target_include_directories(${UnitTest} PRIVATE ${PROJECT_SOURCE_DIR}/ext/base_graph/include)
    add_test(${UnitTest} ${UnitTest})
endforeach()
