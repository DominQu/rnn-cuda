add_library(rnn-neuralnet recurrent.cu)

target_include_directories(rnn-neuralnet PRIVATE ${SOURCE_DIR})