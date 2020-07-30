//
// Created by Jaehee on 2020-07-16.
//

#include "AddingProblem.h"

/// loadVector loads vector from file
/// \param dest [out] double* variable to store the loaded file
/// \param path [in] string variable for the input file path
void loadVector(double *dest, string path) {
    ifstream openFile(path.data());
    if (openFile.is_open()) {
        string line, temp;
        size_t start, end;
        long i = 0;
        while (getline(openFile, line)) {
            dest[i++] = atof(line.c_str());
        }
    } else {
        cout << "Error: cannot read file" << path << endl;
    }
}

/// loadMatrix loads matrix from file
/// \param dest [out] double* variable to store the loaded file
/// \param path [in] string variable for the input file path
void loadMatrix(double **dest, string path) {
    ifstream openFile(path.data());
    if (openFile.is_open()) {
        string line, temp;
        size_t start = 0, end;
        long i = 0; //row
        long j; //column
        while (getline(openFile, line)) {
            j = 0;
            do {
                end = line.find_first_of(',', start);
                temp = line.substr(start, end);
                dest[i][j] = atof(temp.c_str());
                start = end + 1;
                j++;
            } while (start);
            i++;
        }
    } else {
        cout << "Error: cannot read file " << path << endl;
    }
}

/// evalMV computes Matrix-vector multiplication: MV = M cdot V
/// \param MV [in,out] double* vector for the evaluation result to be saved
/// \param M [in] double** matrix for operand. shape: (Mrow, Mcol)
/// \param V [in] double* vector for operand. shape: (Mcol)
/// \param Mrow [in] Long variable that contains the number of rows in M
/// \param Mcol [in] Long variable that contains the number of columns in M
void evalMV(double *MV, double **M, double *V, long Mrow, long Mcol) {
    for (int row = 0; row < Mrow; ++row) {
        MV[row] = 0.;
        for (int col = 0; col < Mcol; ++col) {
            MV[row] += (M[row][col] * V[col]);
        }
    }
}

/// evalAddAndEqual computes in-place vector-vector addition: v1 = v1 + v2
/// \param v1 [in, out] double* variable for operand 1 and storing the evaluation result.
/// \param v2 [in] double* variable for operand 2
/// \param n [in] Long variable that contains the length of both v1 and v2
void evalAddAndEqual(double *v1, double *v2, long n) {
    for (int i = 0; i < n; ++i) {
        v1[i] += v2[i];
    }
}

/// evalAdd computes vector-vector addition: vr = v1 + v2
/// \param vr [out] double* variable to store the evaluation result
/// \param v1 [in] double* variable for operand 1
/// \param v2 [in] double* variable for operand 2
/// \param n [in] Long variable that contains the length of operand vectors
void evalAdd(double *vr, double *v1, double *v2, long n) {
    for (int i = 0; i < n; ++i) {
        vr[i] = v1[i] + v2[i];
    }
}

/// evalMul computes element-wise vector-vector multiplication: vr = v1 * v2
/// \param vr [out] double* variable to store the evaluation result
/// \param v1 [in] double* variable for operand 1
/// \param v2 [in] double* variable for operand 2
/// \param n [in] Long variable that contains the length of operand vectors
void evalMul(double *vr, double *v1, double *v2, long n) {
    for (int i = 0; i < n; ++i) {
        vr[i] = v1[i] * v2[i];
    }
}

/// evalMulAndEqual computes in-place element-wise vector-vector multiplication: v1 = v1 * v2
/// \param v1 [in, out] double* variable for operand 1 and the evaluation result to be stored
/// \param v2 [in] double* variable for operand 2
/// \param n [in] Long variable that contains the length of operand vectors
void evalMulAndEqual(double *v1, double *v2, long n) {
    for (int i = 0; i < n; ++i) {
        v1[i] *= v2[i];
    }
}

/// evalOnem computes vector subtraction from one: vr = 1 - v
/// \param vr [out] double* variable for evaluation result to be stored
/// \param v [in] double* variable for operand
/// \param n [in] Long variable that contains the length of operand vector
void evalOnem(double *vr, double *v, long n) {
    for (int i = 0; i < n; ++i) {
        vr[i] = 1. - v[i];
    }
}

/// approxSigmoid evaluates approximate sigmoid result of input value x: sigmoid(x)
/// \param x [in] double variable for input value x
/// \return [out] approximate sigmoid value of x
double approxSigmoid(double x) {
    double y = sigmoid_coeff[0];

    for (int i = 1; i < 5; i++) {
        y += sigmoid_coeff[i] * pow(x / 8, 2 * i - 1);
    }

    return y;
}

/// approxTanh evaluates approximate tanh result of input value x: tanh(x)
/// \param x [in] double variable for input value x
/// \return [out] approx tanh value of x
double approxTanh(double x) {
    double y = x;

    for (int i = 1; i < 4; i++) {
        y += tanh_coeff[i] * pow(x / 3.46992, 2 * i + 1);
    }

    return y;
}

/// approxTanh evaluates approximate tanh of input vector x
/// \param x [in] double* vector that contains input value x
/// \param n [in] Long variable that contains the input vector length
/// \return [out] approximate tanh result of vector x
double *approxTanh(double *x, long n) {

    double *y = new double[n];
    for (long i = 0; i < n; ++i) {
        y[i] = tanh_coeff[1] * pow(x[i], 3)
               + tanh_coeff[2] * pow(x[i], 5)
               + tanh_coeff[3] * pow(x[i], 7);
    }
    return y;
}

/// wideapprox evaluates approximated nonlinear functions for wider input ranges
/// \param x [in, out] double* vector that contains input vector x and stores result
/// \param length [in] Long variable that contains the length for input vector x
/// \param n, M, L [in] wideapprox related parameters
void wideapprox(double *x, int length, int n, int M, int L) {
    for (long i = 0; i < length; ++i) {
        x[i] = x[i] / (M * pow(L, n - 1));
    }

    for (long i = 0; i < n - 1; ++i) {
        x = approxTanh(x, length);
        for (long j = 0; j < length; j++) {
            x[i] *= L;
        }
    }

    x = approxTanh(x, length);
    for (long i = 0; i < length; i++) {
        x[i] *= M;
    }
}

/// wideApprox evaluates approximated nonlinear functions for wider input ranges
/// \param x [in] double variable that contains input value x
/// \param n, M, L [in] wideapprox related parameters
/// \return [out] wideapprox result of input value x
double wideApprox(double x, double n, double M, double L) {
    double y;
    y = x / (M * pow(L, n - 1));
    for (int i = 0; i < n - 1; i++) {
        y = approxTanh(y) * L;
    }

    y = approxTanh(y) * M;

    return y;
}

/// evalTanh evaluates in-place approximate tanh of input vector x
/// \param x [in, out] double variable that contains input vector x and stores the result
/// \param length [in] Long variable that contains input vector length
void evalTanh(double *x, long length) {
    for (int i = 0; i < length; i++) {
//        x[i] = approxTanh(wideApprox(x[i], 3, 2, 2));
        x[i] = approxTanh(x[i]);
    }
}

/// evalSigmoid evaluates in-place approximate sigmoid of input vector x
/// \param x [in, out] double variable that contains input vector x and stores the result
/// \param length [in] Long variable that contains input vector length
void evalSigmoid(double *x, long length) {
    for (int i = 0; i < length; i++) {
//        x[i] = approxSigmoid(wideApprox(x[i], 3, 4, 2));
        x[i] = approxSigmoid(x[i]);
    }
}

/// printv prints input variable as a vector
/// \param v [in] double variable that contains vector to print
/// \param name [in] string variable that contains name to print
/// \param n [in] Long variable that contains input vector length
void printv(double *v, string name, long n) {
    cout << "-----------" << name << "--------------" << endl;
    double mm = 0.0;
    for (int i = 0; i < n; ++i) {
        cout << v[i] << ",";
        mm = max(mm, abs(v[i]));
    }
    cout << endl;
    cout << "-------------------------------------" << endl;
}

/// printM prints input variable as a matrix
/// \param M [in] double variable that contains matrix to print
/// \param name [in] string variable that contains name to print
/// \param nrow [in] Long variable that contains the number of rows of input matrix
/// \param ncol [in] Long variable that contains the number of cols of input matrix
void printM(double **M, string name, long nrow, long ncol) {
    cout << "-----------" << name << "--------------" << endl;
    for (int j = 0; j < nrow; ++j) {
        for (int i = 0; i < ncol; ++i) {
            cout << M[j][i] << ",";
        }
        cout << endl;
    }
    cout << "-------------------------------------" << endl;

}


AddingProblem::AddingProblem() {}

AddingProblem::AddingProblem(int hiddenSize, int inputSize, int numClass, int bptt) : hiddenSize(hiddenSize),
                                                                                      inputSize(inputSize),
                                                                                      numClass(numClass), bptt(bptt) {
    // initialize weight variables
    this->Wh = new double *[hiddenSize];
    for (int i = 0; i < hiddenSize; ++i) Wh[i] = new double[inputSize]();
    this->Wr = new double *[hiddenSize];
    for (int i = 0; i < hiddenSize; ++i) Wr[i] = new double[inputSize]();
    this->Wz = new double *[hiddenSize];
    for (int i = 0; i < hiddenSize; ++i) Wz[i] = new double[inputSize]();

    this->Uh = new double *[hiddenSize];
    for (int i = 0; i < hiddenSize; ++i) Uh[i] = new double[hiddenSize]();
    this->Ur = new double *[hiddenSize];
    for (int i = 0; i < hiddenSize; ++i) Ur[i] = new double[hiddenSize]();
    this->Uz = new double *[hiddenSize];
    for (int i = 0; i < hiddenSize; ++i) Uz[i] = new double[hiddenSize]();

    this->bh = new double[hiddenSize]();
    this->br = new double[hiddenSize]();
    this->bz = new double[hiddenSize]();

    this->FW = new double *[numClass];
    for (int i = 0; i < numClass; ++i) FW[i] = new double[hiddenSize]();
    this->Fb = new double[numClass]();
}

void AddingProblem::loadWeights(string path) {
    loadMatrix(this->Wh, path + "weights/gru_Wh.csv");
    loadMatrix(this->Wr, path + "weights/gru_Wr.csv");
    loadMatrix(this->Wz, path + "weights/gru_Wz.csv");

    loadMatrix(this->Uh, path + "weights/gru_Uh.csv");
    loadMatrix(this->Ur, path + "weights/gru_Ur.csv");
    loadMatrix(this->Uz, path + "weights/gru_Uz.csv");

    loadVector(this->bh, path + "weights/gru_bh.csv");
    loadVector(this->br, path + "weights/gru_br.csv");
    loadVector(this->bz, path + "weights/gru_bz.csv");

    loadMatrix(this->FW, path + "weights/fc_W.csv");
    loadVector(this->Fb, path + "weights/fc_b.csv");

    cout << "Done loading weights from " << path << endl;
}

void AddingProblem::forward(string input_path) {
    // load input file
    double **sequence = new double *[bptt]; // input sequence, shape: (6, 2)
    for (int i = 0; i < bptt; ++i) sequence[i] = new double[inputSize];
    loadMatrix(sequence, input_path);
    printM(sequence, "input", bptt, 2);

    double **hidden = new double *[bptt + 1]; // hidden state, shape: (6, 64)
    for (int i = 0; i < bptt + 1; ++i) hidden[i] = new double[hiddenSize]();

    // temporary values
    double *Wzx = new double[hiddenSize]();
    double *Uzh = new double[hiddenSize]();
    double *z = new double[hiddenSize]();

    double *Wrx = new double[hiddenSize]();
    double *Urh = new double[hiddenSize]();
    double *r = new double[hiddenSize]();

    double *z1 = new double[hiddenSize]();
    double *zh = new double[hiddenSize]();

    double *Whx = new double[hiddenSize]();
    double *Uhh = new double[hiddenSize]();
    double *g = new double[hiddenSize]();

    double *FWh = new double[numClass]();
    double *output = new double[numClass]();


    /* GRU forward */
    for (int t = 0; t < bptt; ++t) {
        /* r = sigmoid(WrX + UrH + br) */
        evalMV(Wrx, Wr, sequence[t], hiddenSize, inputSize); // Wrx = Wr \cdot X
        evalMV(Urh, Ur, hidden[t], hiddenSize, hiddenSize); // Urh = Ur \cdot H
        evalAdd(r, Urh, Wrx, hiddenSize); // r = WrX + UrH
        evalAddAndEqual(r, br, hiddenSize); // r = WrX + UrH + br

        evalSigmoid(r, hiddenSize); // r = sigmoid(WrX + UrH + br)
//        printv(r, "reset_gate @ step " + to_string(t + 1), hiddenSize);

        /* z = sigmoid(WzX + UzH + bz) */
        evalMV(Wzx, Wz, sequence[t], hiddenSize, inputSize); // Wzx = Wz \cdot X
        evalMV(Uzh, Uz, hidden[t], hiddenSize, hiddenSize); // Uzh = Uz \cdot H
        evalAdd(z, Wzx, Uzh, hiddenSize); // z = WzX + UzH
        evalAddAndEqual(z, bz, hiddenSize); // z = WzX + UzH + bz

        evalSigmoid(z, hiddenSize); // z = sigmoid(WzX + UzH + bz)
//        printv(z, "update_gate @ step " + to_string(t + 1), hiddenSize);

        /* g = tanh(WgX + Ug(r * H) + bg) */
        evalMV(Whx, Wh, sequence[t], hiddenSize, inputSize); // Wgx = Wg \cdot X
        evalMV(Uhh, Uh, hidden[t], hiddenSize, hiddenSize); // Ugh = Ug \cdot X
        evalMul(g, r, Uhh, hiddenSize); // g = UgX * r
        evalAddAndEqual(g, Whx, hiddenSize); // g = WgX + Ug(r * H)
        evalAddAndEqual(g, bh, hiddenSize); // g = WgX + Ug(r * H) + bg
//        printv(g, "sum @ step " + to_string(t + 1), hiddenSize);
        evalTanh(g, hiddenSize); // g = tanh(WgX + Ug(r * H) + bg)
//        printv(g, "new_gate @ step " + to_string(t + 1), hiddenSize);

        /* hidden[t+1] = (1 - z) * g + z * h */
        evalOnem(z1, z, hiddenSize); // z1 = 1 - z
        evalMulAndEqual(g, z1, hiddenSize); // g = (1 -z) * g

        evalMul(zh, z, hidden[t], hiddenSize); // zh = z * hidden[t]
        evalAdd(hidden[t + 1], g, zh, hiddenSize); // hidden[i+1] = (1 - z) * g + z * hidden[t]

        printv(hidden[t + 1], "hidden @ step " + to_string(t + 1), hiddenSize);

    }

    /* fc forward */
    /* output = FWh + Fb */
    evalMV(FWh, FW, hidden[bptt], numClass, hiddenSize);
    evalAdd(output, FWh, Fb, numClass);
    printv(output, "prediction", numClass);

}


int main() {
    int hiddenSize = 64, inputSize = 2, numClass = 1, bptt = 200;
    AddingProblem *model = new AddingProblem(hiddenSize, inputSize, numClass, bptt);
    string path = "/home/hukla/CLionProjects/MHEGRU/addingProblem/";
    model->loadWeights(path);

    model->forward(path + "input_0/input.csv");

    double *originalOutput = new double[numClass];
    loadVector(originalOutput, path + "input_0/output.csv");
    printv(originalOutput, "original output", 1);
}