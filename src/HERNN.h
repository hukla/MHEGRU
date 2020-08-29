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
    double sigmoid3[3] = {0.5,0.1424534,-0.0013186};
    double sigmoid5[4] = {-0.5,0.19131,-0.0045963, 0.0000412332};
    double sigmoid7[5] = {0.5,0.216884,-0.00819276,0.000165861,-0.00000119581};

    // tanh coefficients
    double tanh5[4] = {0.0,0.667391,-0.0440927,0.0010599};
    double tanh_coeff[4] = {1, -8.49814, 11.99804, -6.49478};


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
	void evalTanh(Ciphertext& cipher);

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
