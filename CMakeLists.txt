cmake_minimum_required(VERSION 3.16)
if( CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR )

    project(packed VERSION 0.2.0 LANGUAGES C CXX ASM)

    set(CMAKE_CXX_FLAGS_RELEASE "-O3")

    add_library(packed INTERFACE vector.hpp)
    target_include_directories(packed INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/inc)
    set_target_properties(packed PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED True)

    add_executable(test_vector tests/test_vector.cpp)
    add_executable(test_fft tests/test_fft.cpp)
    add_executable(test_op tests/test_op.cpp)

    target_link_libraries(test_vector PUBLIC packed)
    target_link_libraries(test_fft PUBLIC packed)
    target_link_libraries(test_op PUBLIC packed)

    set_target_properties(test_vector PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED True)
    set_target_properties(test_fft PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED True)
    set_target_properties(test_op PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED True)

    target_compile_options(test_vector PRIVATE -save-temps -march=native)
    target_compile_options(test_fft PRIVATE -save-temps -march=native)
    target_compile_options(test_op PRIVATE -save-temps -ftemplate-backtrace-limit=0 -march=native)
    ENABLE_TESTING()

    add_test(NAME test_vector COMMAND test_vector)
    add_test(NAME test_fft COMMAND test_fft)
    add_test(NAME test_op COMMAND test_op)
else()

endif()
# set(CPACK_PROJECT_NAME ${PROJECT_NAME})
# set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
# include(CPack)
