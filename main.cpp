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

  DataLoader dl("data/dziady-ascii.txt");
  dl.show(std::cout);
  int input_size = dl.getOneHot().getCharacterAmount();

  Recurrent rnn(input_size, 128, 100,-1,1, 0.01);
  // Recurrent rnn(input_size, 128, 100, 0.01, "model1.txt");

  // std::vector<float> loss = rnn.train(10000, dl, 100);
  // rnn.generateText(100, dl, std::cout);
  // rnn.saveModel();
  // float sum = 0;
  // for (auto i:loss) {
  //   sum += i;
  // }
  // std::cout << "Mean loss: " << sum / loss.size() << std::endl;


  return 0;
}
