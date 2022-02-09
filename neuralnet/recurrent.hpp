#pragma once

#include "layers/lstmlayer.hpp"
#include "layers/CCEloss.hpp"
#include "layers/softmax.hpp"
#include "optimizer/sgd.hpp"
#include "loader/loader.hpp"
#include "linalg/GPUMatrix.hpp"
#include <ostream>
#include <string>

class Recurrent {
  private:
  int timesteps;
  LstmLayer lstmlayer1;
  Softmax softmax1;
  CCEloss cceloss1;
  SGD sgd;
  public:
  // Constructor for training a new network
  Recurrent(int input_size, int state_size, 
            int timesteps, float random_weights_low, 
            float random_weights_high, float learning_rate);
  // Constructor for loading trained model
  Recurrent(int input_size, int state_size, 
            int timesteps, float learning_rate, std::string filepath);
  ~Recurrent(){};
  void saveModel(std::string modelname);
  std::vector<float> train(int epochs, DataLoader &dl, int log);
  void test(int generated_text_length, DataLoader &dl);
  void generateText(int generated_text_length, DataLoader &dl, std::ostream &stream);
};