add_library(rnn-layers lstmlayer.cu softmax.cu)

target_include_directories(rnn-layers PRIVATE ${SOURCE_DIR})
