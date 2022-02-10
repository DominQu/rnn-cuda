#include "linalg/GPUMatrix.hpp"
#include "layers/lstmlayer.hpp"
#include "layers/softmax.hpp"
#include "layers/CCEloss.hpp"
#include "loader/loader.hpp"
#include "optimizer/sgd.hpp"
#include "neuralnet/recurrent.hpp"
#include <iostream>

using Matrix = GPUMatrix;

int main() {
  Matrix a = Matrix::from({{ 1, 3 }, { 4, 5 }});
  Matrix b = Matrix::from({{ 1 }, { 1 }});

  std::cout << "A:" << "\n";

  a.show(std::cout);

  std::cout << "\n";

  std::cout << "B:" << "\n";

  b.show(std::cout);

  std::cout << "\n";

  std::cout << "A * B = C:" << "\n";

  Matrix c = a.multiply(b);

  c.show(std::cout);

  std::cout << "\n";

  std::cout << "E" << std::endl;

  Matrix e = Matrix::from({{ 2, 2 }, { 2, 2 }});
  
  e.show(std::cout);

  std::cout << "\n";

  std::cout << "A * E (elementwise) = D:" << std::endl;

  Matrix d = a.multiplyelementwise(e);
  
  d.show(std::cout);

  std::cout << "\n";

  DataLoader dl("data/fakedata-ascii.txt");
  dl.show(std::cout);
  int input_size = dl.getOneHot().getCharacterAmount();
  int state_size = 128;
  int timesteps = 20;
  float learning_rate = 0.01;
  int epochs = 1000;
  int batchsize = 10;
  int log_rate = 50;

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

  // Choose whether you want new network or load network from file
  Recurrent rnn(input_size, state_size, timesteps,-1,1, learning_rate);
  // Recurrent rnn(input_size, state_size, timesteps, learning_rate, name);

  std::vector<float> loss = rnn.train(epochs, batchsize, dl, log_rate);
  rnn.generateText(100, dl, std::cout);
  float sum = 0;
  for (auto i:loss) {
    sum += i;
  }
  std::cout << "Mean loss: " << sum / loss.size() << std::endl;

  Recurrent::saveLoss(loss, name+"_loss");
  rnn.saveModel(name);

  // std::ofstream outfile("fakedata.txt");
  // // outfile.open("fakedata.txt", std::ios::binary);
  // if(!outfile) {
  //   std::cout << "Error\n";
  // }
  // else {
  //   for(int i = 0 ; i < 1000000; i++) {
  //     outfile << "abcdef";
  //   }
  //   outfile.close();
  // }

  return 0;
}
