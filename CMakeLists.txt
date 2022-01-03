cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(RNN LANGUAGES CXX CUDA)

set(SOURCE_FILES test.cu main.cpp)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(rnn ${SOURCE_FILES})
 
set_target_properties(rnn
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
		       CUDA_ARCHITECTURES OFF)
