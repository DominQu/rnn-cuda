add_library(rnn-optimizer sgd.cu rmsprop.cu)

target_include_directories(rnn-optimizer PRIVATE ${SOURCE_DIR})