cmake_minimum_required(VERSION 3.29)
project(cuda CUDA)

set(CMAKE_CUDA_STANDARD 17)
enable_language(CXX CUDA)
add_executable(cuda_01 main.cpp kernel.cu
        main_cl.cpp)
target_include_directories(cuda_01 PUBLIC
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        )
set_target_properties(
        cuda_01
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

#-----------------------------------
find_package(OpenCL)
add_executable(cl_01 main_cl.cpp)
target_link_libraries(cl_01 ${OpenCL_LIBRARIES})

target_include_directories(cl_01 PUBLIC
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)