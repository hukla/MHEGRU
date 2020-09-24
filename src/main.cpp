#include <iostream>
#include "MHEGRU.h"

int main(int argc, char **argv) {
    SetNumThreads(16);
    std::cout << argc << std::endl;
    std::cout << argv[1] << std::endl;
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " TASK_ID" << std::endl;
        std::cerr << "TASK_ID: MNIST=1, addingProblem=2, " << std::endl;
        return 1;
    }

    int hiddenSize, inputSize, numClass, bptt;
    string path;

    if (strcmp(argv[1], "1") == 0)
    {
        // MNIST
        hiddenSize = 64, inputSize = 32, numClass = 16, bptt = 28;
        path = "/home/hukla/PycharmProjects/cryptoGRU/MNIST_enc/";
    } 
    else if (strcmp(argv[1], "2") == 0)
    {
        // addingProblem
        hiddenSize = 64, inputSize = 2, numClass = 1, bptt = 6;
        path = "/home/hukla/CLionProjects/MHEGRU/addingProblem_6/";

    }
    else
    {
        std::cerr << "Usage: " << argv[0] << " TASK_ID" << std::endl;
        std::cerr << "TASK_ID: addingProblem=1, MNIST=2" << std::endl;
        return 1;    
    }
    

    MHEGRU *model = new MHEGRU(hiddenSize, inputSize, numClass, bptt);
    model->loadWeights(path);
    model->encryptWeights();
    // model->printEncryptedWeights();

    model->forward(path + "input_0/");

    return 0;
}
