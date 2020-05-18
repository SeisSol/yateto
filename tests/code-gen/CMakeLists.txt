cmake_minimum_required(VERSION 3.5)
project(unittest)

set(EXAMPLES "matmul;minimal" CACHE STRING "a list of examples to run as tests")
set(SAMPLES hosvd matmul minimal seissol_eqspp stock tce)
set_property(CACHE EXAMPLES PROPERTY STRINGS ${SAMPLES})

set(VARIANT "Eigen" CACHE STRING "example-specific variant")
set(VARIANTS Eigen LIBXSMM OpenBLAS)
set_property(CACHE VARIANT PROPERTY STRINGS ${VARIANTS})

set(PRECISION "double" CACHE STRING "floating point type: double/single")
set(PRECISION_OPTIONS "double" "single")
set_property(CACHE PRECISION PROPERTY STRINGS ${PRECISION_OPTIONS})

set(ARCH "hsw" CACHE STRING "floating point type: double/single")
set(ARCH_OPTIONS snb hsw skx thunderx2t99 nvidia)
set_property(CACHE ARCH PROPERTY STRINGS ${ARCH_OPTIONS})

# adjust cli arg. for the code gen.
if (${PRECISION} STREQUAL "double")
    set(ARCH_STR d${ARCH})
else()
    set(ARCH_STR s${ARCH})
endif()


# find appropriate compiler flags
set(CPU_FLAGS "-march=native" "-mtune=native")
if (${ARCH} STREQUAL "thunderx2t99")
    set(CPU_FLAGS ${CPU_FLAGS} "-mcpu=native")
endif()


find_package(CxxTest REQUIRED)
# attach all necessary external libs
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
if (VARIANT STREQUAL "OpenBLAS")
    find_package(OpenBLAS REQUIRED)
    set(EXTRA_INCLUDES ${OpenBLAS_INCLUDE_DIRS})
    set(EXTRA_LIBS ${OpenBLAS_LIBRARIES})
elseif(VARIANT STREQUAL "LIBSXSMM")
    find_package(Libxsmm_executable REQUIRED)
elseif(VARIANT STREQUAL "Eigen")
    find_package(Eigen3 REQUIRED)
    set(EXTRA_INCLUDES ${EIGEN3_INCLUDE_DIRS})
endif()

enable_testing()

# generate, compile and add numeric tests
foreach(EXAMPLE ${EXAMPLES})
    set(GEN_DIR ${CMAKE_BINARY_DIR}/${EXAMPLE}/${ARCH_STR}_${VARIANT})
    set(GENERATED_SRC ${GEN_DIR}/subroutine.cpp
                      ${GEN_DIR}/init.cpp
                      ${GEN_DIR}/kernel.cpp
                      ${GEN_DIR}/tensor.cpp)

    # execute code generator
    add_custom_command(COMMAND
                        python ${CMAKE_SOURCE_DIR}/generator.py
                        --arch ${ARCH_STR}
                        --variant ${VARIANT}
                        --output_dir ${CMAKE_BINARY_DIR}
                        ${EXAMPLE}
                       WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                       OUTPUT
                        ${GENERATED_SRC}
                        ${GEN_DIR}/init.h
                        ${GEN_DIR}/kernel.h
                        ${GEN_DIR}/tensor.h
                        ${GEN_DIR}/subroutine.h
                        COMMENT "Generating a source code")


    # generate a test runner using CxxTest and a test description
    set(TEST_RUNNER ${CMAKE_BINARY_DIR}/${EXAMPLE}_KernelTest.t.cpp)
    add_custom_command(COMMAND ${CXXTEST_PYTHON_TESTGEN_EXECUTABLE} --error-printer -o ${TEST_RUNNER} ${GEN_DIR}/KernelTest.t.h
                       OUTPUT  ${TEST_RUNNER}
                       DEPENDS ${GEN_DIR}/KernelTest.t.h)

    # compile source code
    add_executable(${EXAMPLE}-target ${TEST_RUNNER} ${GENERATED_SRC})

    # attach necessary include dirs
    target_include_directories(${EXAMPLE}-target PUBLIC ${CMAKE_SOURCE_DIR}/common
                                                      ${CMAKE_SOURCE_DIR}/../../include
                                                      ${GEN_DIR}
                                                      ${CXXTEST_INCLUDE_DIRS}
                                                      ${OpenBLAS_INCLUDE_DIRS})
    target_link_libraries(${EXAMPLE}-target PUBLIC ${EXTRA_LIBS})
    target_compile_options(${EXAMPLE}-target PUBLIC "-std=c++11" ${CPU_FLAGS})
    add_test(${EXAMPLE}-test ${EXAMPLE}-target)
endforeach()