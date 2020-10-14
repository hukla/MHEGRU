#include "MHEGRU.h"
// #define MYDEBUG

// double sigmoid_coeff[5] = {0.5, 1.73496, -4.19407, 5.43402, -2.50739};
// double tanh_coeff[4] = {1, -8.49814, 11.99804, -6.49478};

/// loadVector loads vector from file
/// \param dest [in, out] double* variable to store the loaded file
/// \param path [in] string variable for the input file path
/// \param direction [in] int variable for vector direction 0: col 1: row
void loadVector(double *dest, string path, int direction=0)
{
    ifstream openFile(path.data());
    if (openFile.is_open())
    {
        string line, temp;
        size_t start, end;
        long i = 0;
        if (direction == 1)
        {
            getline(openFile, line);
            start = 0;
            do
            {
                end = line.find_first_of(',', start);
                temp = line.substr(start, end);
                dest[i] = atof(temp.c_str());
                start = end + 1;
                i++;
            } while (start);
        }
        else
        {
            while (getline(openFile, line))
            {
                dest[i++] = atof(line.c_str());
            }
        }
    }
    else
    {
        cout << "Error: cannot read file" << path << endl;
    }
}

/// loadMatrix loads matrix from file
/// \param dest [out] double* variable to store the loaded file
/// \param path [in] string variable for the input file path
void loadMatrix(double **dest, string path)
{
    ifstream openFile(path.data());
    if (openFile.is_open())
    {
        string line, temp;
        size_t start = 0, end;
        long i = 0; //row
        long j;     //column
        while (getline(openFile, line))
        {
            j = 0;
            do
            {
                end = line.find_first_of(',', start);
                temp = line.substr(start, end);
                dest[i][j] = atof(temp.c_str());
                start = end + 1;
                j++;
            } while (start);
            i++;
        }
    }
    else
    {
        cout << "Error: cannot read file " << path << endl;
    }
}
/// printv prints input variable as a vector
/// \param v [in] double variable that contains vector to print
/// \param name [in] string variable that contains name to print
/// \param n [in] Long variable that contains input vector length
void printv(double *v, string name, long n)
{
    cout << "-----------" << name << "--------------" << endl;
    double mm = 0.0;
    for (int i = 0; i < n; ++i)
    {
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
void printM(double **M, string name, long nrow, long ncol)
{
    cout << "-----------" << name << "--------------" << endl;
    for (int j = 0; j < nrow; ++j)
    {
        for (int i = 0; i < ncol; ++i)
        {
            cout << M[j][i] << ",";
        }
        cout << endl;
    }
    cout << "-------------------------------------" << endl;
}

MHEGRU::MHEGRU() { }

MHEGRU::MHEGRU(int hiddenSize, int inputSize, int numClass, int bptt) : hiddenSize(hiddenSize), inputSize(inputSize), numClass(numClass), bptt(bptt)
{
    // generate
    timeutils.start("Generating");
    ring = Ring();
    secretKey = SecretKey(ring);

    //initialize scheme
    scheme = Scheme(secretKey, ring);

    // initialize hernn
    hernn = HERNN(secretKey, scheme);

    scheme.addLeftX0RotKeys(secretKey);
    scheme.addLeftX1RotKeys(secretKey);
    scheme.addBootKey(secretKey, logN0h, logN1, 44);
    timeutils.stop("Generating");

    // ????
    Plaintext diag;
    complex<double> *tmpmsg = new complex<double>[64 * 64]();
    for (int i = 0; i < 64; ++i)
    {
        tmpmsg[i + i * 64] = 1.;
    }
    scheme.encode(diag, tmpmsg, 64, 64, logp);
    ring.rightShiftAndEqual(diag.mx, logQ);
    delete[] tmpmsg;

    // initialize weight variables
    this->Wh = new double *[hiddenSize];
    for (int i = 0; i < hiddenSize; ++i)
        Wh[i] = new double[inputSize]();
    this->Wr = new double *[hiddenSize];
    for (int i = 0; i < hiddenSize; ++i)
        Wr[i] = new double[inputSize]();
    this->Wz = new double *[hiddenSize];
    for (int i = 0; i < hiddenSize; ++i)
        Wz[i] = new double[inputSize]();

    this->Uh = new double *[hiddenSize];
    for (int i = 0; i < hiddenSize; ++i)
        Uh[i] = new double[hiddenSize]();
    this->Ur = new double *[hiddenSize];
    for (int i = 0; i < hiddenSize; ++i)
        Ur[i] = new double[hiddenSize]();
    this->Uz = new double *[hiddenSize];
    for (int i = 0; i < hiddenSize; ++i)
        Uz[i] = new double[hiddenSize]();

    this->bh = new double[hiddenSize]();
    this->br = new double[hiddenSize]();
    this->bz = new double[hiddenSize]();

    this->FW = new double *[numClass];
    for (int i = 0; i < numClass; ++i)
        FW[i] = new double[hiddenSize]();
    this->Fb = new double[numClass]();
}

void MHEGRU::loadWeights(string path) 
{
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

void MHEGRU::encryptWeights() 
{
    timeutils.start("Encrypting Weights");
    hernn.encryptMt(enc_Wh, Wh, hiddenSize, inputSize, logp, logQ);
    hernn.encryptMt(enc_Wz, Wz, hiddenSize, inputSize, logp, logQ);
    hernn.encryptMt(enc_Wr, Wr, hiddenSize, inputSize, logp, logQ);

    hernn.encryptMxt(enc_Uh, Uh, hiddenSize, hiddenSize, logp, logQ);
    hernn.encryptMxt(enc_Uz, Uz, hiddenSize, hiddenSize, logp, logQ);
    hernn.encryptMxt(enc_Ur, Ur, hiddenSize, hiddenSize, logp, logQ);

    hernn.encryptVt(enc_bh, bh, hiddenSize, logp, logQ);
    hernn.encryptVt(enc_bz, bz, hiddenSize, logp, logQ);
    hernn.encryptVt(enc_br, br, hiddenSize, logp, logQ);

    hernn.encryptMx(enc_FW, FW, numClass, hiddenSize, logp, logQ);
    hernn.encryptVx(enc_Fb, Fb, numClass, logp, logQ);
    timeutils.stop("Encrypting Weights");
}

void MHEGRU::printEncryptedWeights()
{
    hernn.printtr(enc_Wh, "Wh");
    hernn.printtr(enc_Wz, "Wz");
    hernn.printtr(enc_Wr, "Wr");

    hernn.printtr(enc_Uh, "Uh");
    hernn.printtr(enc_Uz, "Uz");
    hernn.printtr(enc_Ur, "Ur");

    hernn.print(enc_bh, "bh");
    hernn.print(enc_bz, "bz");
    hernn.print(enc_br, "br");

    hernn.printtr(enc_FW, "FW");
    hernn.print(enc_Fb, "Fb");
}


void MHEGRU::forward(string input_path) 
{
    // load input file
    double **sequence = new double *[bptt]; // input sequence, shape: (bptt, inputSize)
    for (int i = 0; i < bptt; ++i)
        sequence[i] = new double[inputSize];
    loadMatrix(sequence, input_path + "input.csv");

    // addingProblem
    // double *operands = new double[2];
    // int operand_count = 0;
    // for (int i = 0; i < bptt; i++)
    // {
    //     if (sequence[i][1] == 1)
    //     {
    //         operands[operand_count++] = sequence[i][0];
    //     }
    //     if (operand_count == 2)
    //     {
    //         break;
    //     }
    // }
    // printv(operands, "input", 2);
    // printM(sequence, "sequence", 28, 32);

    double **hidden = new double *[bptt + 1]; // hidden state, shape: (bptt + 1, hiddenSize)
    for (int i = 0; i < bptt + 1; ++i)
        hidden[i] = new double[hiddenSize]();

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

    // encrypt input
    Ciphertext enc_x, enc_htr, enc_hidden;

    // intermediate ciphertexts
    Ciphertext enc_Wrx, enc_Urh, enc_Wzx, enc_Uzh, enc_Whx, enc_Uhh, enc_z1, enc_zh, enc_FWh, enc_output;

    // ???????
    Plaintext diag;
    complex<double> *tmpmsg = new complex<double>[64 * 64]();
    for (int i = 0; i < 64; ++i)
    {
        tmpmsg[i + i * 64] = 1.;
    }
    scheme.encode(diag, tmpmsg, 64, 64, logp);
    ring.rightShiftAndEqual(diag.mx, logQ);
    delete[] tmpmsg;
    // ???????

    Ciphertext tmp, tmp2;

    long logq1 = (12 * logp + 2 + 2) + 40;

    hernn.encryptVx(enc_hidden, hidden[0], hiddenSize, logp, logQ);

    double **hidden_plaintext = new double *[bptt + 1]; // hidden state, shape: (bptt + 1, hiddenSize)
    for (int i = 0; i < bptt + 1; ++i)
        hidden_plaintext[i] = new double[hiddenSize]();


    /* GRU forward */
    for (int t = 0; t < bptt; ++t)
    {
        timeutils.start("forward gru step " + to_string(t));

        hernn.encryptVx(enc_x, sequence[t], inputSize, logp, logQ); //(33, 1240)
        hernn.print(enc_x, "input");

        #if defined(MYDEBUG)

        /* r = sigmoid(WrX + UrH + br) */
        hernn.evalMV(enc_Wrx, enc_Wr, enc_x); // Wrx = Wr \cdot X  (33, 1207)
        hernn.printtr(enc_Wrx, "enc_Wrx");
        scheme.modDownTo(tmp, enc_Ur, enc_hidden.logq);
        hernn.printtr(tmp, "enc_Ur mod down");
        hernn.evalMV(enc_Urh, tmp, enc_hidden); // Urh = Ur \cdot H  (33, logq)
        hernn.printtr(enc_Urh, "enc_Urh");
        scheme.modDownToAndEqual(enc_Wrx, enc_Urh.logq);
        hernn.evalAdd(enc_r, enc_Wrx, enc_Urh);    // r = WrX + UrH (33, logq)
        hernn.printtr(enc_r, "enc_r");
        scheme.modDownTo(tmp, enc_br, enc_r.logq); // tmp=enc_br(33, logq)
        hernn.printtr(tmp, "enc_br mod down");
        hernn.evalAddAndEqual(enc_r, tmp);         // r = WrX + UrH + br (33, logq)
        hernn.printtr(enc_r, "enc_r");
        hernn.evalSigmoid(enc_r, 7);               // r = sigmoid(WrX + UrH + br) (33, logq - 4logp - loga)
        hernn.printtr(enc_r, "r gate");

        #else

        /* r = sigmoid(WrX + UrH + br) */
        hernn.evalMV(enc_Wrx, enc_Wr, enc_x); // Wrx = Wr \cdot X  (33, 1207)
        scheme.modDownTo(tmp, enc_Ur, enc_hidden.logq);
        hernn.evalMV(enc_Urh, tmp, enc_hidden); // Urh = Ur \cdot H  (33, logq)
        scheme.modDownToAndEqual(enc_Wrx, enc_Urh.logq);
        hernn.evalAdd(enc_r, enc_Wrx, enc_Urh);    // r = WrX + UrH (33, logq)
        scheme.modDownTo(tmp, enc_br, enc_r.logq); // tmp=enc_br(33, logq)
        hernn.evalAddAndEqual(enc_r, tmp);         // r = WrX + UrH + br (33, logq)
        hernn.evalSigmoid(enc_r, 7);               // r = sigmoid(WrX + UrH + br) (33, logq - 4logp - loga)

        #endif

        /* z = sigmoid(WzX + UzH + bz) */
        hernn.evalMV(enc_Wzx, enc_Wz, enc_x); // Wzx = Wz \cdot X
        // hernn.printtr(enc_Wzx, "enc_Wzx");
        scheme.modDownTo(tmp, enc_Uz, enc_hidden.logq);
        // hernn.printtr(tmp, "tmp");
        hernn.evalMV(enc_Uzh, tmp, enc_hidden); // Uzh = Uz \cdot H
        // hernn.printtr(enc_Uzh, "enc_Uzh");
        scheme.modDownToAndEqual(enc_Wzx, enc_Uzh.logq);
        // hernn.printtr(enc_Wzx, "enc_Wzx");
        hernn.evalAdd(enc_z, enc_Wzx, enc_Uzh); // z = WzX + UzH
        // hernn.printtr(enc_z, "enc_Wzx");
        scheme.modDownTo(tmp, enc_bz, enc_z.logq);
        // hernn.printtr(tmp, "tmp");
        hernn.evalAddAndEqual(enc_z, tmp); // z = WzX + UzH + bz
        // hernn.printtr(enc_z, "enc_z");
        hernn.evalSigmoid(enc_z, 7);       // z = sigmoid(WzX + UzH + bz) (33, logq - 4logp - loga)
        // hernn.printtr(enc_z, "z gate");
        //        printv(z, "update_gate @ step " + to_string(t + 1), hiddenSize);

        /* g = tanh(WgX + Ug(r * H) + bg) */
        hernn.evalMV(enc_Whx, enc_Wh, enc_x); // Wgx = Wg \cdot X
        scheme.modDownTo(tmp, enc_Uh, enc_hidden.logq);
        hernn.evalMV(enc_Uhh, tmp, enc_hidden); // Ugh = Ug \cdot H
        scheme.modDownTo(enc_g, enc_Uhh, enc_r.logq);
        hernn.evalMulAndEqual(enc_g, enc_r); // g = UgH * r
        scheme.modDownToAndEqual(enc_Whx, enc_g.logq);
        hernn.evalAddAndEqual(enc_g, enc_Whx); // g = WgX + Ug(r * H)
        scheme.modDownToAndEqual(enc_bh, enc_g.logq);
        hernn.evalAddAndEqual(enc_g, enc_bh); // g = WgX + Ug(r * H) + bg, enc_g (33,1106)
        hernn.evalTanh(enc_g, 7);                // g = tanh(WgX + Ug(r * H) + bg) enc_g (33, logq - 4logp - loga)
        // hernn.printtr(enc_g, "g gate");

        /* hidden[t+1] = (1 - z) * g + z * h */
        hernn.evalOnem(enc_z1, enc_z, logp);          // z1 = 1 - z (33,1139)
        scheme.modDownToAndEqual(enc_z1, enc_g.logq); // (33, 1005)
        hernn.evalMulAndEqual(enc_g, enc_z1);         // g = (1 -z) * g (33, 972)

        hernn.evalTrx1(enc_htr, enc_hidden, diag); // enc_htr (33, 1207)
        scheme.modDownTo(enc_zh, enc_htr, enc_z.logq);
        hernn.evalMulAndEqual(enc_zh, enc_z);          // zh = z * hidden[t] enc_zh (33,1106)
        scheme.modDownTo(enc_htr, enc_zh, enc_g.logq); // enc_htr (33, 972)
        hernn.evalAddAndEqual(enc_htr, enc_g);         // hidden[i+1] = (1 - z) * g + z * hidden[t]

        // hernn.printtr(enc_htr, "enc_htr");

        if (enc_htr.logq < logq1 && t < bptt - 1)
        {
            //        if(t < bptt - 1) {
            timeutils.start("bootstrap");
            long tmpn0 = enc_htr.n0;
            long tmpn1 = enc_htr.n1;
            enc_htr.n0 = N0h;
            enc_htr.n1 = N1;
            scheme.bootstrapAndEqual(enc_htr, 40, logQ, 4, 4);
            enc_htr.n0 = tmpn0;
            enc_htr.n1 = tmpn1;
            timeutils.stop("bootstrap");
        }
        if (t < bptt - 1)
        {
            hernn.evalTrx2(enc_hidden, enc_htr, diag); // enc_hidden(33, 939)
        }
        timeutils.stop("forward gru step " + to_string(t));
        hernn.printtr(enc_htr, "hidden ciphertext @ gru step " + to_string(t));

        // compare with plaintext hidden
        loadVector(hidden_plaintext[t], input_path + "hidden_" + to_string(t) + ".csv", 1);
        printv(hidden_plaintext[t], "hidden_plaintext", hiddenSize);

        #if defined(MYDEBUG)
        if (t == 2)
        {
            break;
        }
        #endif
    }

    /* fc forward */
    /* output = FWh + Fb */
    timeutils.start("forward fc");
    scheme.modDownTo(tmp, enc_FW, enc_htr.logq);
    hernn.evalMVx(enc_FWh, tmp, enc_htr);
    scheme.modDownTo(enc_output, enc_Fb, enc_FWh.logq);
    hernn.evalAddAndEqual(enc_output, enc_FWh);
    timeutils.stop("forward fc");

    hernn.printResult(enc_output, "Prediction result");
    
}
