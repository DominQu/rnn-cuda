cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(RNN LANGUAGES CXX CUDA)

# Cxx specific
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Cuda specific
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -Wno-deprecated-gpu-targets")
set(CMAKE_CUDA_ARCHITECTURES "35;37;50;52;60;61;70;75;80")
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)

# Common header namespace for all libraries (ex. linalg/matrix.hpp)
include_directories("./.")

add_subdirectory("./linalg")
add_subdirectory("./tests")

add_executable(rnn main.cpp)

target_include_directories(rnn PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(rnn rnn-linalg)

add_compile_options(rnn PUBLIC -g -02)
