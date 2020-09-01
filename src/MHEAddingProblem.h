//
// Created by Jaehee on 2020-07-16.
//

#ifndef MATHEAAN_MHEADDINGPROBLEM_H
#define MATHEAAN_MHEADDINGPROBLEM_H

#include <NTL/ZZ.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <array>
#include <random>
#include <chrono>

#include "MHEAAN/Ring.h"
#include "MHEAAN/SecretKey.h"
#include "MHEAAN/Scheme.h"
#include "MHEAAN/TimeUtils.h"
#include "HERNN.h"

using namespace std;


class MHEAddingProblem {
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

    TimeUtils timeutils;

    /// Encryption
    Ring ring;
    SecretKey secretKey;
    Scheme scheme;
    HERNN hernn;

private:

    long logp = 33;

    /// Ciphertext model weights and intermediate variables
    Ciphertext enc_x, enc_htr, enc_h_l, enc_h_r;
    Ciphertext enc_Wh, enc_Uh, enc_bh; // lr
    Ciphertext enc_Wz, enc_Uz, enc_bz;
    Ciphertext enc_Wr, enc_Ur, enc_br;
    Ciphertext enc_Wzx, enc_Uzh, enc_z;
    Ciphertext enc_Wrx, enc_Urh, enc_r;
    Ciphertext enc_Whx, enc_Uhh, enc_g;

    Ciphertext enc_z1, enc_zh;
    Ciphertext enc_FW, enc_Fb, enc_FWh, enc_output;



public:
    /// default constructor
    MHEAddingProblem();

    /// class constructor
    /// \param hiddenSize [in] integer variable that contains the size of GRU hidden
    /// \param inputSize [in] integer variable that contains the size of model input
    /// \param numClass [in] integer variable that contains the number of classes to classify (1 for regression); the size of model output
    /// \param bptt [in] integer variable that contains the input sequence length
    MHEAddingProblem(int hiddenSize, int inputSize, int numClass, int bptt);

    /// loadWeights loads plaintext weights from file in path
    /// \param path [in] string variable that contains path for model weight files
    void loadWeights(string path);

    void encryptWeights();

    /// forward evaluates model forward propagation
    /// \param input_path [in] string variable that contains path for input string file
    void forward(string input_path);

    /// run MHEAddingProblem
    void run();

};


#endif //MATHEAAN_ADDINGPROBLEM_H
