add_library(rnn-linalg CPUMatrix.cpp GPUMatrix.cu)

target_include_directories(rnn-linalg PRIVATE ${SOURCE_DIR})
