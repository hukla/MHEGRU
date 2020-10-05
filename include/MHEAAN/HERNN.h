#ifndef HERNN_H_
#define HERNN_H_

#include "MHEAAN/SecretKey.h"
#include "MHEAAN/Scheme.h"

using namespace std;

class HERNN {
private:
//    HERNN(SecretKey secretKey, Scheme scheme);

    SecretKey* secretKey;
    Scheme* scheme;

    // sigmoid coefficients
    double sigmoid3[3] = {0.5,0.1424534,-0.0013186}; // x in [-6, 6]
    // double sigmoid5[4] = {-0.5,0.19131,-0.0045963, 0.0000412332}; // x in [-4, 4]
    double sigmoid5[4] = {0.5, 0.19130488174364327, -0.004596051850950526, 4.223017442715702e-05}; // x in [-8, 8]
    // double sigmoid7[5] = {0.5,0.216884,-0.00819276,0.000165861,-0.00000119581}; // x in [-7, 7]
    double sigmoid7[5] = {0.5, 0.21689567455156572, -0.008194757398825834, 0.00016593568955483007, -1.1965564496759948e-06}; // x in [-8, 8]

    // tanh coefficients
    // double tanh5[4] = {0.0,0.667391,-0.0440927,0.0010599}; // x in [-2, 2]
    // double tanh7[5] = {0, 1, -8.49814/pow(3.46992,3), 11.99804/pow(3.46992,5), -6.49478/pow(3.46992,7)};
    // double tanh7[5] = {0, 1, -0.20340681291071229593451461235122, 0.02385135161952251635134053082103, -0.00107232939162246865173849266065}; // x in [-3, 3]

    // TANH COEFFICIENTS in [-4, 4]
    double tanh5[4] = {0.00038260975296624476, 0.7652194684902834, -0.07353682621097166, 0.002702731463794033};  // RMSE: 0.0543
    double tanh7[5] = {0.00043379132689107155, 0.8675825874601593, -0.13111610042441557, 0.010619884719547454, -0.0003063185603617004};  // RMSE: 0.0252

public:
    HERNN(SecretKey& secretKey, Scheme& scheme);

    HERNN();

    void encryptM(Ciphertext& res, double** M, long Mrow, long Mcol, long logp, long logq);
    void encryptMt(Ciphertext& res, double** M, long Mrow, long Mcol, long logp, long logq);

    void encryptMx(Ciphertext& res, double** M, long Mrow, long Mcol, long logp, long logq);
    void encryptMxt(Ciphertext& res, double** M, long Mrow, long Mcol, long logp, long logq);

    void encryptM2(Ciphertext& res1, Ciphertext& res2, double** M, long Mrow, long Mcol, long logp, long logq);
    void encryptMt2(Ciphertext& res1, Ciphertext& res2, double** M, long Mrow, long Mcol, long logp, long logq);

    void encryptV(Ciphertext& res, double* V, long Vrow, long logp, long logq);
    void encryptVt(Ciphertext& res, double* V, long Vrow, long logp, long logq);

    void encryptVx(Ciphertext& res, double* V, long Vrow, long logp, long logq);
    void encryptVxt(Ciphertext& res, double* V, long Vrow, long logp, long logq);

    void encryptV2(Ciphertext& res1, Ciphertext& res2, double* V, long Vrow, long logp, long logq);
    void encryptVt2(Ciphertext& res1, Ciphertext& res2, double* V, long Vrow, long logp, long logq);

    void evalMV(Ciphertext& res, Ciphertext& cipherM, Ciphertext& cipherV);
    void evalMV2(Ciphertext& res, Ciphertext& cipherM1, Ciphertext& cipherM2, Ciphertext& cipherV1, Ciphertext& cipherV2);

    void evalMVx(Ciphertext& res, Ciphertext& cipherM, Ciphertext& cipherV);

    void evalAdd(Ciphertext& res, Ciphertext& cipher1, Ciphertext& cipher2);
    void evalAddAndEqual(Ciphertext& cipher1, Ciphertext& cipher2);
    void evalMul(Ciphertext& res, Ciphertext& cipher1, Ciphertext& cipher2);
    void evalMulAndEqual(Ciphertext& cipher1, Ciphertext& cipher2);
    void evalOnem(Ciphertext& res, Ciphertext& cipher, long logp);
    void evalSigmoid(Ciphertext& cipher);
    void evalTanh(Ciphertext& cipher, int order);

    void evalTrx1(Ciphertext& res, Ciphertext& c, Plaintext& diag);
    void evalTrx2(Ciphertext& res, Ciphertext& c, Plaintext& diag);

    void evalTransp1(Ciphertext& res, Ciphertext& c1, Ciphertext& c2, Plaintext& diag1, Plaintext& diag2);
    void evalTransp2(Ciphertext& res1, Ciphertext& res2, Ciphertext& c, Plaintext& diag1, Plaintext& diag2);

    void print(Ciphertext& x, string name);
    void printtr(Ciphertext& x, string name);
    void prints(Ciphertext& x, string name);

    void evalSigmoid(Ciphertext &cipher, int order);
};

#endif /* HERNN_H_ */
