#include "linalg/GPUMatrix.hpp"
#include "loader/loader.hpp"
#include "neuralnet/recurrent.hpp"
#include <iostream>

using Matrix = GPUMatrix;

void mean(std::vector<MatrixValType> data) {
  float sum = 0;
  for (const auto &i:data) {
    sum += i;
  }
  std::cout << "Mean loss: " << sum / data.size() << std::endl;
}

int main() {

  DataLoader dl("data/fakedata-ascii.txt");
  dl.show(std::cout);

  /// Network and training patameters
  int input_size = dl.getOneHot().getCharacterAmount();
  int state_size = 256;
  int timesteps = 128;
  float learning_rate = 0.001;
  float beta = 0.9;
  float epsilon = 1e-7;
  int epochs = 1000;
  int batchsize = 5;
  int log_rate = 10;
  bool train = true;
  int generated_seq = 100;

  std::string name = "LSTM_epochs_"; 
  name += std::to_string(epochs);
  name += "_batch_";
  name += std::to_string(batchsize);
  name += "_input_";
  name += std::to_string(input_size);
  name += "_state_";
  name += std::to_string(state_size);
  name += "_timesteps_";
  name += std::to_string(timesteps);
  name += "_lr_";
  name += std::to_string(learning_rate);
  std::cout << "Model name: " << name << std::endl;

  if ( train == 1) {
    /// Training new model
    Recurrent rnn(input_size, state_size, timesteps,-1,1, learning_rate, beta, epsilon);
    std::vector<float> loss = rnn.train(epochs, batchsize, dl, log_rate);
    mean(loss);

    Recurrent::saveLoss(loss, name+"_loss");
    rnn.saveModel(name);
    rnn.generateText(generated_seq, dl, std::cout);
  }
  else {
  ///Loading saved model
  Recurrent rnn(input_size, state_size, timesteps, learning_rate, beta, epsilon, name);
  rnn.generateText(100, dl, std::cout);
  }

  return 0;
}
