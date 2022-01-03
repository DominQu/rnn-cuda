cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(RNN LANGUAGES CXX CUDA)

set(SOURCE_FILES matrix.cu main.cpp)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(rnn ${SOURCE_FILES})

add_compile_options(rnn PUBLIC -g -02)

# TODO: nvvc specific options
# add_compile_options(rnn PUBLIC --expt-relaxed-constexpr -Wno-deprecated-gpu-targets --relocatable-device-code true)

set_target_properties(rnn
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
		       CUDA_ARCHITECTURES "35;37;50;52;60;61;70;75;80")
