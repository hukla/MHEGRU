#ifndef RNN_H_
#define RNN_H_

#include <iostream>
// #include "Coefficients.h"

using namespace std;

// static double sigmoid3[3] = {0.5,0.1424534,-0.0013186};
//static double sigmoid5[4] = {0.5,0.1889662,-0.0043265,0.0000368};
//static double sigmoid5[4] = {0.5,0.24938,-0.0193781,0.0011686};
// static double sigmoid7[5] = {0.5,0.2222084,-0.0089966,0.0001929,-0.0000014};

// static double tanh3[3] = {0.0,0.3162573,-0.03196};
//static double tanh5[4] = {0.0,0.4670775,-0.0133625,0.000131213};
//static double tanh5[4] = {0.0,0.9568,-0.2107,0.0235};
// static double tanh5[4] = {0.0,0.667391,-0.0440927,0.0010599};
// static double tanh7[5] = {0.0,0.6268768,-0.0355565,0.0008779,-0.000007};

// sigmoid coefficients
// double sigmoid3[3] = {0.5,0.1424534,-0.0013186}; // x in [-6, 6]
// double sigmoid5[4] = {-0.5,0.19131,-0.0045963, 0.0000412332}; // x in [-4, 4]
// double sigmoid5[4] = {0.5, 0.19130488174364327, -0.004596051850950526, 4.223017442715702e-05}; // x in [-8, 8]
// double sigmoid7[5] = {0.5,0.216884,-0.00819276,0.000165861,-0.00000119581}; // x in [-7, 7]
// double sigmoid7[5] = {0.5, 0.21689567455156572, -0.008194757398825834, 0.00016593568955483007, -1.1965564496759948e-06}; // x in [-8, 8]

// TANH COEFFICIENTS in [-4, 4]
// double tanh5[4] = {0.00038260975296624476, 0.7652194684902834, -0.07353682621097166, 0.002702731463794033};  // RMSE: 0.0543
// double tanh7[5] = {0.00043379132689107155, 0.8675825874601593, -0.13111610042441557, 0.010619884719547454, -0.0003063185603617004};  // RMSE: 0.0252

class RNN {
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
public:
    /// class constructor
    /// \param hiddenSize [in] integer variable that contains the size of GRU hidden
    /// \param inputSize [in] integer variable that contains the size of model input
    /// \param numClass [in] integer variable that contains the number of classes to classify (1 for regression); the size of model output
    /// \param bptt [in] integer variable that contains the input sequence length
    RNN(int hiddenSize, int inputSize, int numClass, int bptt);

	static void readV(double* v, string path);
	static void readVt(double* v, string path);
	static void readM(double** M, string path);

	static void evalMV(double* MV, double** M, double* V, long Mrow, long Mcol);
	static void evalMVx(double* MV, double** M, double* V, long Mrow, long Mcol);
	static void evalAdd(double* vr, double* v1, double* v2, long n);
	static void evalAddAndEqual(double* v1, double* v2, long n);
	static void evalMul(double* vr, double* v1, double* v2, long n);
	static void evalMulAndEqual(double* v1, double* v2, long n);
	static void evalOnem(double* vr, double* v, long n);

	static void evalSigmoid(double* x, long n);
	static void evalSigmoid(double* x, long n, int order);
	static void evalTanh(double* x, long n);
	static void evalTanh(double* x, long n, int order);

	static void printv(double* v, string name, long n);
	static void printM(double** M, string name, long nrow, long ncol);

};

#endif
