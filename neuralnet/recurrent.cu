#include "recurrent.hpp"

Recurrent::Recurrent(int input_size, int state_size, 
                     int timesteps, float random_weights_low, 
                     float random_weights_high, float learning_rate, 
                     float beta, float epsilon) 
                     : input_size(input_size),
                       state_size(state_size),
                       timesteps(timesteps),
                       lstmlayer1({input_size, state_size, timesteps, 
                                   random_weights_low, random_weights_high}), 
                       softmax1({input_size}), 
                       cceloss1({}), 
                       sgd({learning_rate}),
                       rms({learning_rate, beta, input_size, state_size, epsilon}){ }
Recurrent::Recurrent(int input_size, int state_size, 
                     int timesteps, float learning_rate, float beta, float epsilon, std::string filepath) 
                     : input_size(input_size),
                       state_size(state_size),
                       timesteps(timesteps),
                       lstmlayer1({input_size, state_size, 
                                   timesteps, filepath}), 
                       softmax1({input_size}), 
                       cceloss1({}), 
                       sgd({learning_rate}),
                       rms({learning_rate, beta, input_size, state_size, epsilon}) { }

std::vector<float> Recurrent::train(int epochs, DataLoader &dl, int log) {
  std::vector<float> loss;

  for(int epoch = 0; epoch < epochs; epoch++) {
    // get training batch and separate the label
    std::vector<GPUMatrix> batch = dl.getTrainSequence(this->timesteps+1, 1);
    GPUMatrix label = batch.back();
    batch.pop_back();  

    GPUMatrix lstm_output = this->lstmlayer1.forward(batch);
    GPUMatrix softmax_output = this->softmax1.forward(lstm_output);
    MatrixValType cceloss_output = this->cceloss1.forward(softmax_output, label);

    if(epoch % log == 0) {
        std::cout << "Epoch: " << epoch;
        std::cout << ", Current loss: " << cceloss_output;
        std::cout << "\n"; 
    }

    loss.push_back(cceloss_output);

    GPUMatrix cceloss_gradient = cceloss1.backward(softmax_output, label);
    std::vector<GPUMatrix> lstm_gradient = lstmlayer1.backward(cceloss_gradient, batch);
    std::vector<GPUMatrix> optimizer_output = rms.calculateUpdate(lstm_gradient);
    lstmlayer1.updateWeights(optimizer_output);
  }
  return loss;
}

std::vector<float> Recurrent::train(int epochs, int batchsize, DataLoader &dl, int log) {
  std::vector<float> loss;

  GPUMatrix gradient_input_f(MatrixSize(this->state_size, this->input_size), 0);
  GPUMatrix gradient_input_g(MatrixSize(this->state_size, this->input_size), 0);
  GPUMatrix gradient_input_i(MatrixSize(this->state_size, this->input_size), 0);
  GPUMatrix gradient_input_o(MatrixSize(this->state_size, this->input_size), 0);

  GPUMatrix gradient_state_f(MatrixSize(this->state_size, this->state_size), 0);
  GPUMatrix gradient_state_g(MatrixSize(this->state_size, this->state_size), 0);
  GPUMatrix gradient_state_i(MatrixSize(this->state_size, this->state_size), 0);
  GPUMatrix gradient_state_o(MatrixSize(this->state_size, this->state_size), 0);

  GPUMatrix gradient_input_bias_f(MatrixSize(this->state_size, 1), 0);
  GPUMatrix gradient_input_bias_g(MatrixSize(this->state_size, 1), 0);
  GPUMatrix gradient_input_bias_i(MatrixSize(this->state_size, 1), 0);
  GPUMatrix gradient_input_bias_o(MatrixSize(this->state_size, 1), 0);

  GPUMatrix gradient_output_bias(MatrixSize(this->input_size, 1), 0);
  GPUMatrix gradient_output_weights(MatrixSize(this->input_size, this->state_size), 0);
  auto training_start = std::chrono::high_resolution_clock::now(); 

  for(int epoch = 0; epoch < epochs; epoch++) {
    auto epoch_start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<GPUMatrix>> gradients;
    MatrixValType batch_loss = 0;
    for(int b = 0; b < batchsize; b++) {
      // get training batch and separate the label
      std::vector<GPUMatrix> batch = dl.getTrainSequence(this->timesteps+1, 1);
      GPUMatrix label = batch.back();
      batch.pop_back();  
  
      std::vector<GPUMatrix> lstm_output = this->lstmlayer1.forward(batch, 1);
      std::vector<GPUMatrix> softmax_output = this->softmax1.forward(lstm_output, 1);
      std::vector<MatrixValType> cceloss_output = this->cceloss1.forward(softmax_output, label, batch);
      for(auto &i:cceloss_output) {
        batch_loss += i; 
      }
  
      std::vector<GPUMatrix> cceloss_gradient = cceloss1.backward(softmax_output, label, batch);
      std::vector<GPUMatrix> lstm_gradient = lstmlayer1.backward(cceloss_gradient, batch);

      gradients.push_back(lstm_gradient);
    }
    for(const auto &i:gradients) {
      gradient_input_f.add(i[0], gradient_input_f);
      gradient_input_g.add(i[1], gradient_input_g);
      gradient_input_i.add(i[2], gradient_input_i);
      gradient_input_o.add(i[3], gradient_input_o);
    
      gradient_state_f.add(i[4], gradient_state_f);
      gradient_state_g.add(i[5], gradient_state_g);
      gradient_state_i.add(i[6], gradient_state_i);
      gradient_state_o.add(i[7], gradient_state_o);
    
      gradient_output_weights.add(i[8], gradient_output_weights);
    
      gradient_input_bias_f.add(i[9], gradient_input_bias_f);
      gradient_input_bias_g.add(i[10], gradient_input_bias_g);
      gradient_input_bias_i.add(i[11], gradient_input_bias_i);
      gradient_input_bias_o.add(i[12], gradient_input_bias_o);
    
      gradient_output_bias.add(i[13], gradient_output_bias);

    }
    MatrixValType scalar = 1.f/(float)batchsize;
    gradient_input_f.multiply(scalar, gradient_input_f);
    gradient_input_g.multiply(scalar, gradient_input_g);
    gradient_input_i.multiply(scalar, gradient_input_i);
    gradient_input_o.multiply(scalar, gradient_input_o);
  
    gradient_state_f.multiply(scalar, gradient_state_f);
    gradient_state_g.multiply(scalar, gradient_state_g);
    gradient_state_i.multiply(scalar, gradient_state_i);
    gradient_state_o.multiply(scalar, gradient_state_o);
  
    gradient_output_weights.multiply(scalar, gradient_output_weights);
  
    gradient_input_bias_f.multiply(scalar, gradient_input_bias_f);
    gradient_input_bias_g.multiply(scalar, gradient_input_bias_g);
    gradient_input_bias_i.multiply(scalar, gradient_input_bias_i);
    gradient_input_bias_o.multiply(scalar, gradient_input_bias_o);
  
    gradient_output_bias.multiply(scalar, gradient_output_bias);
    std::vector<GPUMatrix> mean_gradients;
    mean_gradients.push_back(gradient_input_f);
    mean_gradients.push_back(gradient_input_g);
    mean_gradients.push_back(gradient_input_i);
    mean_gradients.push_back(gradient_input_o);

    mean_gradients.push_back(gradient_state_f);
    mean_gradients.push_back(gradient_state_g);
    mean_gradients.push_back(gradient_state_i);
    mean_gradients.push_back(gradient_state_o);

    mean_gradients.push_back(gradient_output_weights);

    mean_gradients.push_back(gradient_input_bias_f);
    mean_gradients.push_back(gradient_input_bias_g);
    mean_gradients.push_back(gradient_input_bias_i);
    mean_gradients.push_back(gradient_input_bias_o);
    mean_gradients.push_back(gradient_output_bias);

    std::vector<GPUMatrix> optimizer_output = rms.calculateUpdate(mean_gradients);
    lstmlayer1.updateWeights(optimizer_output);
    
    auto batch_end = std::chrono::high_resolution_clock::now();
    loss.push_back(batch_loss / (batchsize * this->timesteps));
    if(epoch % log == 0) {
        std::cout << "Epoch: " << epoch;
        std::cout << ", Current loss: " << batch_loss / (batchsize * this->timesteps);
        auto durationsec1 = std::chrono::duration_cast<std::chrono::seconds>(batch_end - epoch_start);
        auto durationmilli1 = std::chrono::duration_cast<std::chrono::milliseconds>(batch_end - epoch_start);
        std::cout << ", Epoch time: ";
        std::cout << durationsec1.count() << ".";
        std::cout << durationmilli1.count() - 1000*durationsec1.count() << " sec";
        std::cout << "\n"; 
    }

  }

  auto training_stop = std::chrono::high_resolution_clock::now();

  auto traininghour = std::chrono::duration_cast<std::chrono::hours>(training_stop - training_start); 
  auto trainingmin = std::chrono::duration_cast<std::chrono::minutes>(training_stop - training_start);
  auto trainingsec = std::chrono::duration_cast<std::chrono::seconds>(training_stop - training_start);

  std::cout << "Training duration: ";
  std::cout << traininghour.count() << "h ";
  std::cout << trainingmin.count() - 60 * traininghour.count() << "min ";
  std::cout << trainingsec.count() - 60 * trainingmin.count() << "sec\n";

  return loss;
}

float Recurrent::test( DataLoader &dl) {
  std::vector<GPUMatrix> start = dl.getTestBatch();
  std::vector<GPUMatrix> batch;
  int tests = start.size() - (this->timesteps + 1);
  if( tests > 100) {
    tests = 100;
  }
  int correct = 0;

  for(int t = 0;t < tests; t++) {
    batch = std::vector<GPUMatrix>(start.begin() + t, start.begin() + this->timesteps + t);
    GPUMatrix label = start[this->timesteps + t + 1];

    GPUMatrix lstm_output = lstmlayer1.forward(start);  
    GPUMatrix softmax_output = softmax1.forward(lstm_output);
    CPUMatrix probabilities = softmax_output.toCPU();

    int pred_argmax = probabilities.argmax();
    int real_argmax = label.toCPU().argmax();
    if(pred_argmax == real_argmax) {
      correct += 1;
    }

  }

  float accuracy = correct / tests;

  std::cout << "Test accuracy: " << accuracy << std::endl;
  return accuracy;

}

void Recurrent::generateText(int generated_text_length, DataLoader &dl, std::ostream &stream) {
  // Show text given to the network as input
  std::vector<GPUMatrix> start = dl.getTrainBatch(this->timesteps);
  for( auto i:start) {
      stream << dl.getOneHot().decode(i.toCPU());
  }
  stream << "\n --------\n";
  // Generate text of given length
  for( int letter = 0; letter < generated_text_length; letter++) {

    GPUMatrix lstm_output = lstmlayer1.forward(start);  
    GPUMatrix softmax_output = softmax1.forward(lstm_output);
    CPUMatrix probabilities = softmax_output.toCPU();

    int argmax = probabilities.argmax();
    CPUMatrix onehot(probabilities.getSize(), 0);
    onehot.at(argmax, 0) = 1.0f;  
    GPUMatrix nextletter = GPUMatrix::from(onehot);

    // Append network input with predicted letter
    start.push_back(nextletter);
    start = std::vector<GPUMatrix>(start.begin() + 1, start.end());

    // Show generated letters to output stream
    stream << dl.getOneHot().decode(onehot);
  }
  std::cout << "\n"; 
}

void Recurrent::saveModel(std::string modelname) {
  std::fstream outFile;
  outFile.open(modelname, std::ios::out);
  if (!outFile) {
      std::cout << "Can't open file " << modelname << std::endl;
  }
  else {
      this->lstmlayer1.saveWeights(outFile);
      std::cout << "Model saved succesfully" << std::endl;
      outFile.close();
  }
}

void Recurrent::saveLoss(std::vector<MatrixValType> loss, std::string filename) {
  std::fstream outFile;
  outFile.open(filename, std::ios::out);
  if (!outFile) {
      std::cout << "Can't open file " << filename << std::endl;
  }
  else {
      for(auto &val:loss) {
        outFile << std::to_string(val) << "\n";
      }
      std::cout << "Loss saved succesfully" << std::endl;
      outFile.close();
  }
}