#include <iostream>
#include "MHEGRU.h"

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " TASK_ID MODEL_PATH INPUT_PATH" << std::endl;
        std::cerr << "TASK_ID : MNIST=1, addingProblem=2, " << std::endl;
        std::cerr << "MODEL_PATH : Path to model weights" << std::endl;
        std::cerr << "INPUT_PATH : Path to input files" << std::endl;
        return 1;
    }

    int hiddenSize, inputSize, numClass, bptt;
    string model_path, input_path;

    SetNumThreads(16);

    if (strcmp(argv[1], "1") == 0)
    {
        // MNIST
        hiddenSize = 64, inputSize = 32, numClass = 16, bptt = 28; // actual inputSize=28, numClass=10
        if (argc == 4)
        {
            model_path = argv[2];
            input_path = argv[3];
        }
        else
        {
            model_path = "MNIST/";
            input_path = "MNIST/input_0";
        }
    }
    else if (strcmp(argv[1], "2") == 0)
    {
        // addingProblem
        hiddenSize = 64, inputSize = 2, numClass = 1, bptt = 6;
        if (argc == 4)
        {
            model_path = argv[2];
            input_path = argv[3];
        }
        else
        {
            model_path = "addingProblem_6/";
            input_path = "addingProblem_6/input_0";
        }
    }
    else
    {
        std::cerr << "Invalid TASK_ID" << argv[2] << std::endl;
        std::cerr << "TASK_ID: addingProblem=1, MNIST=2" << std::endl;
        return 1;
    }

    MHEGRU *model = new MHEGRU(hiddenSize, inputSize, numClass, bptt);
    model->loadWeights(model_path);
    model->encryptWeights();
    // model->printEncryptedWeights();

    model->forward(input_path);

    return 0;
}
