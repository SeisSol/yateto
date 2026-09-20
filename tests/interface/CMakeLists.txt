cmake_minimum_required(VERSION 3.5)
project(inteface-test)

find_package(CxxTest REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()

set(TEST_HEADERS TensorView.t.h Type.t.h)
set(TEST_HEADER_PATHS)
foreach(header ${TEST_HEADERS})
    list(APPEND TEST_HEADER_PATHS ${CMAKE_SOURCE_DIR}/${header})
endforeach()

# generate and add an interface test
add_custom_command(COMMAND ${CXXTEST_PYTHON_TESTGEN_EXECUTABLE}
                           --error-printer -o interface.t.cpp ${TEST_HEADER_PATHS}
                   OUTPUT  interface.t.cpp
                   DEPENDS ${TEST_HEADER_PATHS}
                   COMMENT "Generating a test runner")

add_executable(interface-target interface.t.cpp)
target_include_directories(interface-target PUBLIC ${CMAKE_SOURCE_DIR}/../../include ${CXXTEST_INCLUDE_DIRS})
add_test(interface interface-target)
