cmake_minimum_required(VERSION 3.5)
project(inteface-test)

find_package(CxxTest REQUIRED)

enable_testing()
# generate and add an interface test
 add_custom_command(COMMAND ${CXXTEST_PYTHON_TESTGEN_EXECUTABLE} 
                            --error-printer -o TensorView.t.cpp ${CMAKE_SOURCE_DIR}/TensorView.t.h
                    OUTPUT  TensorView.t.cpp
                    COMMENT "Generating a test runner")

add_executable(tensor-view-target TensorView.t.cpp)
target_include_directories(tensor-view-target PUBLIC ${CMAKE_SOURCE_DIR}/../../include ${CXXTEST_INCLUDE_DIRS})
target_compile_options(tensor-view-target PUBLIC "-std=c++11")
add_test(tensor-view tensor-view-target)