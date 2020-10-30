#ifndef __MHEGRU_H__
#define __MHEGRU_H__

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

#include "MHEAAN/TimeUtils.h"
#include "MHEAAN/Ring.h"
#include "MHEAAN/SecretKey.h"
#include "MHEAAN/Scheme.h"
#include "MHEAAN/HERNN.h"

class MHEGRU {
private:

    // the size of GRU hidden
    int hiddenSize;
    // the size of model input
    int inputSize;
    // the number of classes to classify (1 for regression); the size of model output
    int numClass;
    // the input sequence length
    int bptt;

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

    long logp = 33;

    /// Ciphertext model weights and intermediate variables
    Ciphertext enc_Wh, enc_Uh, enc_bh; // lr
    Ciphertext enc_Wz, enc_Uz, enc_bz;
    Ciphertext enc_Wr, enc_Ur, enc_br;
    Ciphertext enc_z, enc_r, enc_g;

    Ciphertext enc_FW, enc_Fb;


public:
    /// default constructor
    MHEGRU();

    /// class constructor
    /// \param hiddenSize [in] integer variable that contains the size of GRU hidden
    /// \param inputSize [in] integer variable that contains the size of model input
    /// \param numClass [in] integer variable that contains the number of classes to classify (1 for regression); the size of model output
    /// \param bptt [in] integer variable that contains the input sequence length
    MHEGRU(int hiddenSize, int inputSize, int numClass, int bptt);

    /// loadWeights loads plaintext weights from file in path
    /// \param path [in] string variable that contains path for model weight files
    void loadWeights(string path);

    void encryptWeights();

    void printEncryptedWeights();

    /// forward evaluates plaintext model forward propagation
    /// \param input_path [in] string variable that contains path for input string file
    void forwardPlx(string input_path);

    /// forward evaluates model forward propagation
    /// \param input_path [in] string variable that contains path for input string file
    void forward(string input_path);


};

#endif // __MHEGRU_H__