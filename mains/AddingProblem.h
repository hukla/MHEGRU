//
// Created by Jaehee on 2020-07-16.
//

#ifndef MATHEAAN_ADDINGPROBLEM_H
#define MATHEAAN_ADDINGPROBLEM_H

#include <NTL/ZZ.h>

#include <math.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <array>
#include <random>
#include <chrono>

using namespace std;

double sigmoid_coeff[5] = {0.5, 1.73496, -4.19407, 5.43402, -2.50739};
double tanh_coeff[4] = {1, -8.49814, 11.99804, -6.49478};

class AddingProblem {
private:

    // the size of GRU hidden
    int hiddenSize;
    // the size of model input
    int inputSize;
    // the number of classes to classify (1 for regression); the size of model output
    int numClass;
    // the input sequence length
    int bptt;

    /// AddingProblem model weights
    // GRU weights for input to hidden
    double **Wh, **Wr, **Wz;
    // GRU weights for hidden to hidden
    double **Uh, **Ur, **Uz;
    // GRU biases
    double *bh, *br, *bz;
    // FC layer weight
    double** FW;
    // FC layer bias
    double* Fb;

public:
    /// default constructor
    AddingProblem();

    /// class constructor
    /// \param hiddenSize [in] integer variable that contains the size of GRU hidden
    /// \param inputSize [in] integer variable that contains the size of model input
    /// \param numClass [in] integer variable that contains the number of classes to classify (1 for regression); the size of model output
    /// \param bptt [in] integer variable that contains the input sequence length
    AddingProblem(int hiddenSize, int inputSize, int numClass, int bptt);

    /// loadWeights loads weights from file in path
    /// \param path [in] string variable that contains path for model weight files
    void loadWeights(string path);

    /// forward evaluates model forward propagation
    /// \param input_path [in] string variable that contains path for input string file
    void forward(string input_path);

};


#endif //MATHEAAN_ADDINGPROBLEM_H
