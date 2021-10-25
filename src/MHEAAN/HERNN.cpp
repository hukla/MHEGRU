#include "HERNN.h"
#include "Scheme.h"
#include "Coefficients.h"

HERNN::HERNN(SecretKey &secretKey, Scheme &scheme) : secretKey(&secretKey), scheme(&scheme)
{
}

void HERNN::encryptM(Ciphertext &res, double **M, long Mrow, long Mcol, long logp, long logq)
{
    double *tmp = new double[Mrow * Mcol]();
    for (int i = 0; i < Mrow; ++i)
    {
        for (int j = 0; j < Mcol; ++j)
        {
            tmp[i + j * Mrow] = M[i][j];
        }
    }
    scheme->encrypt(res, tmp, Mrow, Mcol, logp, logq);
    delete[] tmp;
}

void HERNN::encryptMt(Ciphertext &res, double **M, long Mrow, long Mcol, long logp, long logq)
{
    double *tmp = new double[Mrow * Mcol]();
    for (int i = 0; i < Mrow; ++i)
    {
        for (int j = 0; j < Mcol; ++j)
        {
            tmp[j + i * Mcol] = M[i][j];
        }
    }
    scheme->encrypt(res, tmp, Mcol, Mrow, logp, logq);
    delete[] tmp;
}

void HERNN::encryptMx(Ciphertext &res, double **M, long Mrow, long Mcol, long logp, long logq)
{
    long Mrow2 = 1 << (long)ceil(log2(Mrow));
    long Mcol2 = 1 << (long)ceil(log2(Mcol));
    double *tmp = new double[Mrow2 * Mcol2]();
    for (int i = 0; i < Mrow; ++i)
    {
        for (int j = 0; j < Mcol; ++j)
        {
            tmp[i + j * Mrow2] = M[i][j];
        }
    }
    scheme->encrypt(res, tmp, Mrow2, Mcol2, logp, logq);
    delete[] tmp;
}

void HERNN::encryptMxt(Ciphertext &res, double **M, long Mrow, long Mcol, long logp, long logq)
{
    long Mrow2 = 1 << (long)ceil(log2(Mrow));
    long Mcol2 = 1 << (long)ceil(log2(Mcol));
    double *tmp = new double[Mrow2 * Mcol2]();
    for (int i = 0; i < Mrow; ++i)
    {
        for (int j = 0; j < Mcol; ++j)
        {
            tmp[j + i * Mcol2] = M[i][j];
        }
    }
    scheme->encrypt(res, tmp, Mcol2, Mrow2, logp, logq);
    delete[] tmp;
}

void HERNN::encryptM2(Ciphertext &res1, Ciphertext &res2, double **M, long Mrow, long Mcol, long logp, long logq)
{
    double *tmp = new double[Mrow * Mcol / 2]();
    for (int i = 0; i < Mrow / 2; ++i)
    {
        for (int j = 0; j < Mcol; ++j)
        {
            tmp[i + j * Mrow / 2] = M[i][j];
        }
    }
    scheme->encrypt(res1, tmp, Mrow / 2, Mcol, logp, logq);
    for (int i = 0; i < Mrow / 2; ++i)
    {
        for (int j = 0; j < Mcol; ++j)
        {
            tmp[i + j * Mrow / 2] = M[i + Mrow / 2][j];
        }
    }
    scheme->encrypt(res2, tmp, Mrow / 2, Mcol, logp, logq);
    delete[] tmp;
}

void HERNN::encryptMt2(Ciphertext &res1, Ciphertext &res2, double **M, long Mrow, long Mcol, long logp, long logq)
{
    double *tmp = new double[Mrow * Mcol / 2]();
    for (int i = 0; i < Mrow; ++i)
    {
        for (int j = 0; j < Mcol / 2; ++j)
        {
            tmp[j + i * Mcol / 2] = M[i][j];
        }
    }
    scheme->encrypt(res1, tmp, Mcol / 2, Mrow, logp, logq);
    for (int i = 0; i < Mrow; ++i)
    {
        for (int j = 0; j < Mcol / 2; ++j)
        {
            tmp[j + i * Mcol / 2] = M[i][j + Mcol / 2];
        }
    }
    scheme->encrypt(res2, tmp, Mcol / 2, Mrow, logp, logq);
    delete[] tmp;
}

void HERNN::encryptV(Ciphertext &res, double *V, long Vrow, long logp, long logq)
{
    scheme->encrypt(res, V, Vrow, 1, logp, logq);
}

void HERNN::encryptVt(Ciphertext &res, double *V, long Vrow, long logp, long logq)
{
    scheme->encrypt(res, V, 1, Vrow, logp, logq);
}

void HERNN::encryptVx(Ciphertext &res, double *V, long Vrow, long logp, long logq)
{
    long Vrow2 = 1 << (long)ceil(log2(Vrow));
    double *V2 = new double[Vrow2]();
    for (int i = 0; i < Vrow; ++i)
    {
        V2[i] = V[i];
    }
    scheme->encrypt(res, V2, Vrow2, 1, logp, logq);
    delete[] V2;
}

void HERNN::encryptVxt(Ciphertext &res, double *V, long Vrow, long logp, long logq)
{
    long Vrow2 = 1 << (long)ceil(log2(Vrow));
    double *V2 = new double[Vrow2]();
    for (int i = 0; i < Vrow; ++i)
    {
        V2[i] = V[i];
    }
    scheme->encrypt(res, V2, 1, Vrow2, logp, logq);
    delete[] V2;
}

void HERNN::encryptV2(Ciphertext &res1, Ciphertext &res2, double *V, long Vrow, long logp, long logq)
{
    scheme->encrypt(res1, V, Vrow / 2, 1, logp, logq);
    scheme->encrypt(res2, (V + (Vrow / 2)), Vrow / 2, 1, logp, logq);
}

void HERNN::encryptVt2(Ciphertext &res1, Ciphertext &res2, double *V, long Vrow, long logp, long logq)
{
    scheme->encrypt(res1, V, 1, Vrow / 2, logp, logq);
    scheme->encrypt(res2, (V + (Vrow / 2)), 1, Vrow / 2, logp, logq);
}

void HERNN::evalMV(Ciphertext &res, Ciphertext &cipherM, Ciphertext &cipherV)
{
    time_t start, end;
    start = time(NULL);
    scheme->mult(res, cipherM, cipherV);
    end = time(NULL);

    printf("mult: %f", (double)(end - start));
    Ciphertext rot;
    for (int i = 1; i < cipherM.n0; i <<= 1)
    {
        scheme->leftRotate(rot, res, i, 0);
        scheme->addAndEqual(res, rot);
    }
    scheme->reScaleByAndEqual(res, cipherM.logp);
    res.n0 = 1;
    res.n1 = cipherM.n1;
}

void HERNN::evalMV2(Ciphertext &res, Ciphertext &cipherM1, Ciphertext &cipherM2, Ciphertext &cipherV1, Ciphertext &cipherV2)
{
    Ciphertext tmp;
    evalMV(res, cipherM1, cipherV1);
    evalMV(tmp, cipherM2, cipherV2);
    scheme->addAndEqual(res, tmp);
}

void HERNN::evalMVx(Ciphertext &res, Ciphertext &cipherM, Ciphertext &cipherV)
{
    scheme->mult(res, cipherM, cipherV);
    Ciphertext rot;
    for (int i = 1; i < cipherM.n1; i <<= 1)
    {
        scheme->leftRotate(rot, res, 0, i);
        scheme->addAndEqual(res, rot);
    }
    scheme->reScaleByAndEqual(res, cipherM.logp);
    res.n0 = cipherM.n0;
    res.n1 = 1;
}

void HERNN::evalAdd(Ciphertext &res, Ciphertext &cipher1, Ciphertext &cipher2)
{
    scheme->add(res, cipher1, cipher2);
}

void HERNN::evalAddAndEqual(Ciphertext &cipher1, Ciphertext &cipher2)
{
    scheme->addAndEqual(cipher1, cipher2);
}

void HERNN::evalMul(Ciphertext &res, Ciphertext &cipher1, Ciphertext &cipher2)
{
    scheme->mult(res, cipher1, cipher2);
    scheme->reScaleByAndEqual(res, cipher1.logp);
}

void HERNN::evalMulAndEqual(Ciphertext &cipher1, Ciphertext &cipher2)
{
    scheme->multAndEqual(cipher1, cipher2);
    scheme->reScaleByAndEqual(cipher1, cipher2.logp);
}

void HERNN::evalOnem(Ciphertext &res, Ciphertext &cipher, long logp)
{
    scheme->negate(res, cipher);
    scheme->addConstAndEqual(res, 1.0, logp);
}

void HERNN::evalSigmoid(Ciphertext &cipher)
{
    long logp = cipher.logp;
    long loga = 2;							 //TODO: why???
    scheme->reScaleByAndEqual(cipher, loga); // cipher(31, 1205)

    /* 3-order approximation */
    Ciphertext enca2;
    enca2.copy(cipher);
    scheme->squareAndEqual(enca2);												 // enca2(62, 1205)
    scheme->reScaleByAndEqual(enca2, logp);										 // enca2(29, 1172)
    scheme->addConstAndEqual(enca2, sigmoid3[1] / sigmoid3[2], logp - 2 * loga); // enca2(29, 1172), logp - 2 * loga = 29
    scheme->multConstAndEqual(cipher, sigmoid3[2], logp + 3 * loga);			 // cipher(31, 1205) -> (70, 1205)
    scheme->reScaleByAndEqual(cipher, logp);									 // cipher(37, 1172)
    scheme->multAndEqual(cipher, enca2);										 // cipher(66, 1172)
    scheme->reScaleByAndEqual(cipher, logp);									 // cipher(33,1139)
    scheme->addConstAndEqual(cipher, sigmoid3[0]);								 // cipher(33,1139)

}

void HERNN::evalSigmoid(Ciphertext &cipher, int order)
{
    long logp = cipher.logp;
    long loga = 2;							 // why???
    scheme->reScaleByAndEqual(cipher, loga); // cipher(logp-loga, logq-loga)

    Ciphertext encIP2; /// x^2
    scheme->square(encIP2, cipher);
    scheme->reScaleByAndEqual(encIP2, logp); // encIP2(logp-2loga, logq-logp-loga)

    if (order == 7)
    {
        /* 7-order approximation */
        Ciphertext encIP4;						 /// x^4
        scheme->square(encIP4, encIP2);			 // encIP4(2logp-4loga, logq-logp-loga)
        scheme->reScaleByAndEqual(encIP4, logp); // encIP4(logp-4loga, logq-2logp-loga)

        Ciphertext encIP2c;																/// a3/a4 x^2
        scheme->multConst(encIP2c, encIP2, sigmoid7[3] / sigmoid7[4], logp - 2 * loga); // encIP2c(2logp-4loga, logq-logp-loga)
        scheme->reScaleByAndEqual(encIP2c, logp);										// encIP2c(logp-4loga, logq-2logp-loga)

        scheme->addAndEqual(encIP4, encIP2c);										  // encIP4(logp-4loga, logq-2logp-loga)
        scheme->addConstAndEqual(encIP4, sigmoid7[2] / sigmoid7[4], logp - 4 * loga); // encIP4(logp-4loga, logq-2logp-loga)
                                                                                      // encIP4 = x^4 + a3/a4 x^2 + a2/a4

        Ciphertext tmp0; /// a1 x + a0
        scheme->multConst(tmp0, cipher, sigmoid7[1], logp + loga); // tmp0(2logp, logq-loga)
        scheme->reScaleByAndEqual(tmp0, logp); // tmp0(logp, logq-logp-loga)
        scheme->addConstAndEqual(tmp0, sigmoid7[0], logp); //tmp0(logp, logq-logp-loga)

        scheme->multConstAndEqual(cipher, sigmoid7[4], logp + 7 * loga); // tmp1(2logp+6loga, logq-loga)
        scheme->reScaleByAndEqual(cipher, logp); // tmp1(logp+6loga, logq-logp-loga)
        scheme->multAndEqual(cipher, encIP2); // tmp1(2logp+4loga, logq-logp-loga)
        scheme->reScaleByAndEqual(cipher, logp); // tmp1(logp+4loga, logq-2logp-loga)

        scheme->multAndEqual(cipher, encIP4); // tmp1(2logp, logq-2logp-loga)
        scheme->reScaleByAndEqual(cipher, logp); // tmp1(logp, logq-3logp-loga)
        scheme->modDownToAndEqual(tmp0, cipher.logq); // tmp0(logp, logq-3logp-loga)
        scheme->addAndEqual(cipher, tmp0); // tmp1(logp, logq-3logp-loga)

    }
    else if (order == 5)
    {
        /* 5-order approximation */
        Ciphertext enca2c, encac;
        enca2c.copy(encIP2);
        scheme->addConstAndEqual(enca2c, sigmoid5[2] / sigmoid5[3], logp - 2 * loga); // x^2 + c[2] / c[3]
        scheme->multConst(encac, cipher, sigmoid5[1], logp + loga);					  // c[1] * x
        scheme->multConstAndEqual(cipher, sigmoid5[3], logp + 5 * loga);			  // c[3] * x
        scheme->reScaleByAndEqual(cipher, logp);
        scheme->multAndEqual(cipher, encIP2);  // c[3]x * x^2
        scheme->multAndEqual(cipher, enca2c); // c[3]x * x^2 * (x^2 + c[2] / c[3])
        scheme->reScaleByAndEqual(cipher, 2 * logp);
        scheme->reScaleByAndEqual(encac, logp);
        scheme->modDownByAndEqual(encac, 2 * logp);
        scheme->addAndEqual(cipher, encac);
        scheme->addConstAndEqual(cipher, sigmoid5[0]);
    }
    else
    {
        evalSigmoid(cipher);
    }
}

void HERNN::evalTanh(Ciphertext &cipher, int order)
{ // cipher(logp,loga)
    long logp = cipher.logp;
    long loga = 2;
    scheme->reScaleByAndEqual(cipher, loga); // cipher(logp-loga,logq-loga)

    Ciphertext encIP2; // x^2
    encIP2.copy(cipher);
    scheme->squareAndEqual(encIP2);			// encIP2(2logp-2loga,logq-loga)
    scheme->reScaleByAndEqual(encIP2, logp); // encIP2(logp-2loga,logq-logp-loga)

    if(order == 7){
        /* 7-order approximation */
        Ciphertext encIP4;						 /// x^4
        scheme->square(encIP4, encIP2);			 // encIP4(2logp-4loga, logq-logp-loga)
        scheme->reScaleByAndEqual(encIP4, logp); // encIP4(logp-4loga, logq-2logp-loga

        Ciphertext encIP2c;																/// a3/a4 x^2
        scheme->multConst(encIP2c, encIP2, tanh7[3] / tanh7[4], logp - 2 * loga); // encIP2c(2logp-4loga, logq-logp-loga)
        scheme->reScaleByAndEqual(encIP2c, logp);										// encIP2c(logp-4loga, logq-2logp-loga)

        scheme->addAndEqual(encIP4, encIP2c);										  // encIP4(logp-4loga, logq-2logp-loga)
        scheme->addConstAndEqual(encIP4, tanh7[2] / tanh7[4], logp - 4 * loga); // encIP4(logp-4loga, logq-2logp-loga)
                                                                                      // encIP4 = x^4 + a3/a4 x^2 + a2/a4

        Ciphertext tmp0; /// a1 x + a0
        scheme->multConst(tmp0, cipher, tanh7[1], logp + loga); // tmp0(2logp, logq-loga)
        scheme->reScaleByAndEqual(tmp0, logp); // tmp0(logp, logq-logp-loga)
        scheme->addConstAndEqual(tmp0, tanh7[0], logp); //tmp0(logp, logq-logp-loga)

        scheme->multConstAndEqual(cipher, tanh7[4], logp + 7 * loga); // tmp1(2logp+6loga, logq-loga)
        scheme->reScaleByAndEqual(cipher, logp); // tmp1(logp+6loga, logq-logp-loga)
        scheme->multAndEqual(cipher, encIP2); // tmp1(2logp+4loga, logq-logp-loga)
        scheme->reScaleByAndEqual(cipher, logp); // tmp1(logp+4loga, logq-2logp-loga)

        scheme->multAndEqual(cipher, encIP4); // tmp1(2logp, logq-2logp-loga)
        scheme->reScaleByAndEqual(cipher, logp); // tmp1(logp, logq-3logp-loga)
        scheme->modDownToAndEqual(tmp0, cipher.logq); // tmp0(logp, logq-3logp-loga)
        scheme->addAndEqual(cipher, tmp0); // tmp1(logp, logq-3logp-loga)


    } else if (order == 5) {
        /* 5-order approximation */
        Ciphertext enca2c, encac; //x^2+c[2]/c[3], c[1]x
        enca2c.copy(encIP2);
        scheme->addConstAndEqual(enca2c, tanh5[2] / tanh5[3], logp - 2 * loga); // enca2c(logp-2loga,logq-logp-loga)
        scheme->multConst(encac, cipher, tanh5[1], logp + loga);				// encac(2logp, logq-loga)
        scheme->multConstAndEqual(cipher, tanh5[3], logp + 5 * loga);			// cipher(2logp+4loga,logq-loga)
        scheme->reScaleByAndEqual(cipher, logp);								// cipher(logp+4loga,logq-logp-loga)
        scheme->multAndEqual(cipher, encIP2);									// cipher(2logp+2loga, logq-logp-loga)
        scheme->multAndEqual(cipher, enca2c);									// cipher(3logp, logq-logp-loga)
        scheme->reScaleByAndEqual(cipher, 2 * logp);							// cipher(logp, logq-3logp-loga)
        scheme->reScaleByAndEqual(encac, logp);									// encac(logp, logq-logp-loga)
        scheme->modDownByAndEqual(encac, 2 * logp);								// encac(logp, logq-3logp-loga)
        scheme->addAndEqual(cipher, encac);										// cipher (logp, logq-3logp-loga)
    } else {
        Ciphertext enca2;
        enca2.copy(cipher);
        scheme->squareAndEqual(enca2);												 // enca2(62, 1205)
        scheme->reScaleByAndEqual(enca2, logp);										 // enca2(29, 1172)
        scheme->addConstAndEqual(enca2, tanh3[1] / tanh3[2], logp - 2 * loga); // enca2(29, 1172), logp - 2 * loga = 29
        scheme->multConstAndEqual(cipher, tanh3[2], logp + 3 * loga);			 // cipher(31, 1205) -> (70, 1205)
        scheme->reScaleByAndEqual(cipher, logp);									 // cipher(37, 1172)
        scheme->multAndEqual(cipher, enca2);										 // cipher(66, 1172)
        scheme->reScaleByAndEqual(cipher, logp);									 // cipher(33,1139)
        scheme->addConstAndEqual(cipher, tanh3[0]);								 // cipher(33,1139)
    }
}

void HERNN::evalTrx1(Ciphertext &res, Ciphertext &c, Plaintext &diag)
{
    scheme->multPoly(res, c, diag.mx, diag.logp);

    Ciphertext tmp;
    for (int i = 1; i < 64; i <<= 1)
    {
        scheme->leftRotate(tmp, res, i, 0);
        scheme->addAndEqual(res, tmp);
    }
    scheme->reScaleByAndEqual(res, diag.logp);
    res.n0 = 1;
    res.n1 = 64;
}

void HERNN::evalTrx2(Ciphertext &res, Ciphertext &c, Plaintext &diag)
{
    Ciphertext tmp;
    scheme->multPoly(res, c, diag.mx, diag.logp);

    for (int i = 1; i < 64; i <<= 1)
    {
        scheme->leftRotate(tmp, res, 0, i);
        scheme->addAndEqual(res, tmp);
    }
    scheme->reScaleByAndEqual(res, diag.logp);
    res.n0 = 64;
    res.n1 = 1;
}

void HERNN::evalTransp1(Ciphertext &res, Ciphertext &c1, Ciphertext &c2, Plaintext &diag1, Plaintext &diag2)
{
    Ciphertext tmp;
    scheme->multPoly(tmp, c1, diag1.mx, diag1.logp);
    scheme->multPoly(res, c2, diag2.mx, diag2.logp);
    scheme->addAndEqual(res, tmp);

    for (int i = 1; i < N0h; i <<= 1)
    {
        scheme->leftRotate(tmp, res, i, 0);
        scheme->addAndEqual(res, tmp);
    }
    scheme->reScaleByAndEqual(res, diag1.logp);
    res.n0 = 1;
    res.n1 = N1;
}

void HERNN::evalTransp2(Ciphertext &res1, Ciphertext &res2, Ciphertext &c, Plaintext &diag1, Plaintext &diag2)
{
    Ciphertext tmp;
    scheme->multPoly(res1, c, diag1.mx, diag1.logp);
    scheme->multPoly(res2, c, diag2.mx, diag2.logp);

    for (int i = 1; i < N1; i <<= 1)
    {
        scheme->leftRotate(tmp, res1, 0, i);
        scheme->addAndEqual(res1, tmp);
        scheme->leftRotate(tmp, res2, 0, i);
        scheme->addAndEqual(res2, tmp);
    }
    scheme->reScaleByAndEqual(res1, diag1.logp);
    scheme->reScaleByAndEqual(res2, diag2.logp);
    res1.n0 = N0h;
    res1.n1 = 1;
    res2.n0 = N0h;
    res2.n1 = 1;
}

void HERNN::print(Ciphertext &x, string name)
{
    cout << "-----------" << name << "--------------" << endl;
    cout << "slots:" << x.n0 << ", " << x.n1 << endl;
    cout << "bits:" << x.logp << ", " << x.logq << endl;
    complex<double> *vals = scheme->decrypt(*secretKey, x);
    double mm = 0.0;
    int label = 0;
    cout << "values:";
    for (int j = 0; j < x.n1; ++j)
    {
        for (int i = 0; i < x.n0; ++i)
        {
            cout << vals[i + j * x.n0].real() << ",";
            mm = max(mm, vals[i + j * x.n0].real());
            if (mm == vals[i + j * x.n0].real())
            {
                label = i;
            }
        }
        cout << endl;
    }
    cout << "max:" << mm << endl;
    cout << "-------------------------------------" << endl;
    delete[] vals;
}

void HERNN::printResult(Ciphertext &x, string name)
{
	cout << "-----------" << name << "--------------" << endl;
	cout << "slots:" << x.n0 << ", " << x.n1 << endl;
	cout << "bits:" << x.logp << ", " << x.logq << endl;
	complex<double> *vals = scheme->decrypt(*secretKey, x);
	double mm = 0.0;
	int label = 0;
	cout << "prediction values:";
	for (int j = 0; j < x.n1; ++j)
	{
		for (int i = 0; i < x.n0; ++i)
		{
			cout << vals[i + j * x.n0].real() << ",";
			mm = max(mm, vals[i + j * x.n0].real());
			if (mm == vals[i + j * x.n0].real())
			{
				label = i;
			}
		}
		cout << endl;
	}
	cout << "max:" << mm << endl;
	cout << "regression result:" << mm << endl;
	cout << "argmax result:" << label << endl;
	cout << "-------------------------------------" << endl;
	delete[] vals;
}

void HERNN::printtr(Ciphertext &x, string name)
{
    cout << "-----------" << name << "--------------" << endl;
    cout << "slots:" << x.n0 << ", " << x.n1 << endl;
    cout << "bits:" << x.logp << ", " << x.logq << endl;
    complex<double> *vals = scheme->decrypt(*secretKey, x);
    double mm = 0.0;
    cout << "values:";
    for (int i = 0; i < x.n0; ++i)
    {
        for (int j = 0; j < x.n1; ++j)
        {
            cout << vals[i + j * x.n0].real() << ",";
            mm = max(mm, vals[i + j * x.n0].real());
        }
        cout << endl;
    }
    cout << "max:" << mm << endl;
    cout << "-------------------------------------" << endl;
    delete[] vals;
}

void HERNN::prints(Ciphertext &x, string name)
{
    cout << "-----------" << name << "--------------" << endl;
    cout << "slots: " << x.n0 << ", " << x.n1 << endl;
    cout << "bits: " << x.logp << ", " << x.logq << endl;
    cout << "-------------------------------------" << endl;
}

HERNN::HERNN()
{
}
