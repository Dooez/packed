cmake_minimum_required(VERSION 3.16)
if( CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR )

    project(packed VERSION 0.2.0 LANGUAGES C CXX ASM)

    set(CMAKE_CXX_FLAGS_RELEASE "-O3")

    add_library(packed INTERFACE vector.hpp)
    target_include_directories(packed INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
    set_target_properties(packed PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED True)

    add_executable(test_vector tests/test_vector.cpp)
    target_link_libraries(test_vector PUBLIC packed)
    set_target_properties(test_vector PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED True)

    ENABLE_TESTING()

    add_test(NAME test_vector COMMAND test_vector)
else()

endif()
# set(CPACK_PROJECT_NAME ${PROJECT_NAME})
# set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
# include(CPack)
