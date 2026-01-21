# Copyright 2020 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.13)

# Defines
set(SNRT_NFPU_PER_CORE "0" CACHE STRING "Number of FPUs per Spatz")

list(APPEND CMAKE_MODULE_PATH ${SPATZD_ROOT}/sw/cmake)

message(STATUS "SPATZ_ROOT: ${SPATZD_ROOT}")

set(CMAKE_TOOLCHAIN_FILE toolchain-llvm)

project(Benchmarks LANGUAGES C ASM)
include(SnitchUtilities)

# Build the runtime.
add_subdirectory(${SPATZD_ROOT}/sw/snRuntime snRuntime)

include_directories(${SPATZD_ROOT}/sw/include)
include_directories(${SNRUNTIME_INCLUDE_DIRS})

message(STATUS "COMMON_SRCS: ${COMMON_SRCS}")
message(STATUS "COMMON_INCS: ${COMMON_INCS}")

add_compile_options(-O3 -g -ffunction-sections -D__spatz__ -Dhalf=_Float16 ${COMMON_INCS})

# Add the main executable.

file(GLOB SPATZ_RUNTIME_MAIN "${CMAKE_CURRENT_SOURCE_DIR}/codegen/*/src/*_offload_runtime_main.spatz.c")

add_snitch_executable(main 
    ${SPATZ_RUNTIME_MAIN}
    ${COMMON_SRCS}
)

# Link with runtime if needed.
target_link_libraries(main ${SNITCH_RUNTIME})
target_compile_definitions(main PUBLIC SNRT_NFPU_PER_CORE=${SNRT_NFPU_PER_CORE})

# Optionally, add include directories for your own headers
target_include_directories(main PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
