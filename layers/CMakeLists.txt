add_library(rnn-layers linearlayer.cu 
                       sigmoidlayer.cu
                       tanhlayer.cu
                       lstmlayer.cu)

target_include_directories(rnn-layers PRIVATE ${SOURCE_DIR})
