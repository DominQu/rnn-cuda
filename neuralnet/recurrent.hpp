#pragma once

#include "layers/lstmlayer.hpp"
#include "layers/CCEloss.hpp"
#include "layers/softmax.hpp"
#include "optimizer/sgd.hpp"
#include "optimizer/rmsprop.hpp"
#include "loader/loader.hpp"
#include "linalg/GPUMatrix.hpp"
#include <ostream>
#include <string>
#include <chrono>

class Recurrent {
  private:
  int input_size;
  int state_size;
  int timesteps;
  LstmLayer lstmlayer1;
  Softmax softmax1;
  CCEloss cceloss1;
  SGD sgd;
  RMSprop rms;
  public:
  /// Constructor for training a new network with rmsprop optimizer
  Recurrent(int input_size, int state_size, 
            int timesteps, float random_weights_low, 
            float random_weights_high, float learning_rate, 
            float beta, float epsilon);
  /// Constructor for loading trained model with rmsptop optimizer
  Recurrent(int input_size, int state_size, 
            int timesteps, float learning_rate, float beta, float epsilon, std::string filepath);
  ~Recurrent() {};
  void saveModel(std::string modelname);
  static void saveLoss(std::vector<MatrixValType> loss, std::string filename);
  /// Train model using stochastic gradient descent
  std::vector<float> train(int epochs, DataLoader &dl, int log);
  /// Train model using mini-batch gradient descent
  std::vector<float> train(int epochs, int batchsize, DataLoader &dl, int log);
  float test( DataLoader &dl);
  void generateText(int generated_text_length, DataLoader &dl, std::ostream &stream);
};