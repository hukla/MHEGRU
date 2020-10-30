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
    void printResult(Ciphertext& x, string name);

    void evalSigmoid(Ciphertext &cipher, int order);
};

#endif /* HERNN_H_ */
