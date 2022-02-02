cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(RNN LANGUAGES CXX CUDA)

# Cxx specific
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Cuda specific
find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_80,code=sm_80 --expt-relaxed-constexpr -Wno-deprecated-gpu-targets")
set(CMAKE_CUDA_ARCHITECTURES OFF)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)

# Common header namespace for all libraries (ex. linalg/matrix.hpp)
include_directories("./.")

add_subdirectory("./linalg")
add_subdirectory("./loader")
add_subdirectory("./layers")
add_subdirectory("./tests")

add_executable(rnn main.cpp)

target_include_directories(rnn PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(rnn rnn-linalg rnn-layers)

add_compile_options(rnn PUBLIC -g -02)

