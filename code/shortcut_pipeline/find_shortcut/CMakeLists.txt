cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_C_COMPILER AND EXISTS "/usr/bin/gcc-12")
    set(CMAKE_C_COMPILER "/usr/bin/gcc-12")
endif()
if(NOT DEFINED CMAKE_CXX_COMPILER AND EXISTS "/usr/bin/g++-12")
    set(CMAKE_CXX_COMPILER "/usr/bin/g++-12")
endif()
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND EXISTS "/usr/bin/g++-12")
    set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++-12")
endif()

project(find_shortcut_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES "80" CACHE STRING "CUDA architectures" FORCE)

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
find_path(JANSSON_INCLUDE_DIR NAMES jansson.h)
find_library(JANSSON_LIBRARY NAMES jansson)

if(NOT JANSSON_INCLUDE_DIR OR NOT JANSSON_LIBRARY)
    message(FATAL_ERROR "jansson headers/libraries not found. Install libjansson-dev (or equivalent).")
endif()

add_executable(cuda test.cu)
target_include_directories(cuda PRIVATE ${JANSSON_INCLUDE_DIR})
target_link_libraries(cuda PRIVATE CUDA::cudart ${JANSSON_LIBRARY} Threads::Threads)
target_compile_options(cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

message(STATUS "CUDA toolkit root: ${CUDAToolkit_ROOT}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Jansson include dir: ${JANSSON_INCLUDE_DIR}")
message(STATUS "Jansson library: ${JANSSON_LIBRARY}")
