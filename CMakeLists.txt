cmake_minimum_required(VERSION 3.29)
project(cuda CUDA)

set(CMAKE_CUDA_STANDARD 17)
enable_language(CXX CUDA)
add_executable(cuda_01 main.cpp kernel.cu)
target_include_directories(cuda_01 PUBLIC
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        )
set_target_properties(
        cuda_01
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)