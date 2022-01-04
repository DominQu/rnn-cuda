cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(RNN LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# TODO: nvvc specific options
# add_compile_options(rnn PUBLIC --expt-relaxed-constexpr -Wno-deprecated-gpu-targets --relocatable-device-code true)

add_library(rnn-lib matrix.cu)
target_include_directories(rnn-lib PUBLIC ${PROJECT_SOURCE_DIR})

set_target_properties(rnn-lib
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
		       CUDA_ARCHITECTURES "35;37;50;52;60;61;70;75;80")

add_subdirectory("./tests")
		     
add_executable(rnn main.cpp)
target_link_libraries(rnn rnn-lib)
add_compile_options(rnn PUBLIC -g -02)


