#include <iostream>
#include <fstream>
#include "MHEGRU.h"
#include "TestScheme.h"
#include <cxxopts.hpp>

int test() {

//----------------------------------------------------------------------------------
//   STANDARD TESTS
//----------------------------------------------------------------------------------

	TestScheme::testEncrypt(300, 30, 2, 2);
	TestScheme::testStandard(300, 30, 2, 2);
	TestScheme::testimult(300, 30, 2, 2);

	TestScheme::testRotateFast(300, 30, 3, 3, 1, 0);
	TestScheme::testConjugate(300, 30, 2, 2);

	TestScheme::testBootstrap(40, 33, 7, 8, 4, 4);

	return 0;
}

int main(int argc, const char **argv)
{
    cxxopts::Options options("MHEGRU");
    options.add_options()
        ("d,debug", "Enable debugging", cxxopts::value<bool>()->default_value("false"))
        ("h,help", "Print usage")
        ("t,task", "TASK_ID : MNIST=1, addingProblem=2, ", cxxopts::value<int>()->default_value("1"))
        ("m,model", "MODEL_PATH : Path to model weights", cxxopts::value<std::string>())
        ("i,input", "INPUT_PATH : Path to input files", cxxopts::value<std::string>())
        ("n,thread", "THREAD NUM", cxxopts::value<int>()->default_value("16"))
    ;

    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
      std::cout << options.help() << std::endl;
      exit(0);
    }

    bool debug = result["debug"].as<bool>();

    int hiddenSize, inputSize, numClass, bptt;
    string model_path, input_path;

    int num_threads = result["thread"].as<int>();
    int task_id = result["task"].as<int>();

    SetNumThreads(num_threads);


    if (task_id == 1)
    {
        // MNIST
        hiddenSize = 64, inputSize = 32, numClass = 16, bptt = 28; // actual inputSize=28, numClass=10
        if (result.count("model"))
        {
            model_path = result["model"].as<string>();
        }
        else
        {
            model_path = "MNIST/";
        }

        if (result.count("input"))
        {
            input_path = result["input"].as<string>();
        }
        else
        {
            input_path = "MNIST/input_0/";
        }
    }
    else if (task_id == 2)
    {
        // addingProblem
        hiddenSize = 64, inputSize = 2, numClass = 1, bptt = 6;
        if (result.count("model"))
        {
            model_path = result["model"].as<string>();
        }
        else
        {
            model_path = "addingProblem_6/";
        }

        if (result.count("input"))
        {
            input_path = result["input"].as<string>();
        }
        else
        {
            input_path = "addingProblem_6/input_0/";
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
    model->forwardPlx(input_path);
    model->encryptWeights();

    if (debug)
    {
        model->profile(input_path);
    }
    else
    {
        try
        {
            model->forward(input_path);
        } catch (const exception e) {
            cout << e.what() << endl;
        }
    }

    return 0;
}
