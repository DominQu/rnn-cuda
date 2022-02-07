add_library(rnn-layers lstmlayer.cu softmax.cu CCEloss.cu)

target_include_directories(rnn-layers PRIVATE ${SOURCE_DIR})
