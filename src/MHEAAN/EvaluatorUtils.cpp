#include "EvaluatorUtils.h"


//----------------------------------------------------------------------------------
//   RANDOM REAL & COMPLEX
//----------------------------------------------------------------------------------


double EvaluatorUtils::randomReal(double bound)  {
	return (double) rand()/(RAND_MAX) * bound;
}

double EvaluatorUtils::randomRealSigned(double bound)  {
	return ((double)rand()/(RAND_MAX) * 2.0 - 1.0) * bound;
}

complex<double> EvaluatorUtils::randomComplex(double bound) {
	complex<double> res;
	res.real(randomReal(bound));
	res.imag(randomReal(bound));
	return res;
}

complex<double> EvaluatorUtils::randomComplexSigned(double bound) {
	complex<double> res;
	res.real(randomRealSigned(bound));
	res.imag(randomRealSigned(bound));
	return res;
}

complex<double> EvaluatorUtils::randomCircle(double anglebound) {
	double angle = randomReal(anglebound);
	complex<double> res;
	res.real(cos(angle * 2 * M_PI));
	res.imag(sin(angle * 2 * M_PI));
	return res;
}

double* EvaluatorUtils::randomRealArray(long n, double bound) {
	double* res = new double[n];
	for (long i = 0; i < n; ++i) {
		res[i] = randomReal(bound);
	}
	return res;
}

double* EvaluatorUtils::randomRealSignedArray(long n, double bound) {
	double* res = new double[n];
	for (long i = 0; i < n; ++i) {
		res[i] = randomRealSigned(bound);
	}
	return res;
}

complex<double>* EvaluatorUtils::randomComplexArray(long n, double bound) {
	complex<double>* res = new complex<double>[n];
	for (long i = 0; i < n; ++i) {
		res[i] = randomComplex(bound);
	}
	return res;
}

complex<double>* EvaluatorUtils::randomComplexSignedArray(long n, double bound) {
	complex<double>* res = new complex<double>[n];
	for (long i = 0; i < n; ++i) {
		res[i] = randomComplexSigned(bound);
	}
	return res;
}

complex<double>* EvaluatorUtils::randomCircleArray(long n, double bound) {
	complex<double>* res = new complex<double>[n];
	for (long i = 0; i < n; ++i) {
		res[i] = randomCircle(bound);
	}
	return res;
}


//----------------------------------------------------------------------------------
//   DOUBLE & RR <-> ZZ
//----------------------------------------------------------------------------------


double EvaluatorUtils::scaleDownToReal(ZZ& x, long logp) {
	RR xp = to_RR(x);
	xp.e -= logp;
	return to_double(xp);
}

ZZ EvaluatorUtils::scaleUpToZZ(double x, long logp) {
	RR rx = to_RR(x);
	return scaleUpToZZ(rx, logp);
}

ZZ EvaluatorUtils::scaleUpToZZ(RR& x, long logp) {
	RR xp = MakeRR(x.x, x.e + logp);
	return RoundToZZ(xp);
}


//----------------------------------------------------------------------------------
//   ROTATIONS
//----------------------------------------------------------------------------------


void EvaluatorUtils::leftRotateAndEqual(complex<double>* vals, long n0, long n1, long r0, long r1) {
	r0 %= n0;
	if(r0 != 0) {
		long divisor = GCD(r0, n0);
		long steps = n0 / divisor;
		for (long i = 0; i < n1; ++i) {
			for (long j = 0; j < divisor; ++j) {
				complex<double> tmp = vals[j + i * n0];
				long idx = j;
				for (long k = 0; k < steps - 1; ++k) {
					vals[idx + i * n0] = vals[((idx + r0) % n0) + i * n0];
					idx = (idx + r0) % n0;
				}
				vals[idx + i * n0] = tmp;
			}
		}
	}
	r1 %= n1;
	if(r1 != 0) {
		long divisor = GCD(r1, n1);
		long steps = n1 / divisor;
		for (long i = 0; i < n0; ++i) {
			for (long j = 0; j < divisor; ++j) {
				complex<double> tmp = vals[i + j * n0];
				long idy = j;
				for (long k = 0; k < steps - 1; ++k) {
					vals[i + idy * n0] = vals[i + ((idy + r1) % n1) * n0];
					idy = (idy + r1) % n1;
				}
				vals[i + idy * n0] = tmp;
			}
		}
	}
}

void EvaluatorUtils::rightRotateAndEqual(complex<double>* vals, long n0, long n1, long r0, long r1) {
	r0 %= n0;
	r0 = (n0 - r0) % n0;
	r1 %= n1;
	r1 = (n1 - r1) % n1;
	leftRotateAndEqual(vals, n0, n1, r0, r1);
}


//----------------------------------------------------------------------------------
//   MATRIX
//----------------------------------------------------------------------------------


complex<double>* EvaluatorUtils::transpose(complex<double>* vals, long n) {
	complex<double>* res = new complex<double>[n * n];
	for (long i = 0; i < n; ++i) {
		for (long j = 0; j < n; ++j) {
			res[i + j * n] = vals[j + i * n];
		}
	}
	return res;
}

complex<double>* EvaluatorUtils::squareMatMult(complex<double>* vals1, complex<double>* vals2, long n) {
	complex<double>* res = new complex<double>[n * n];
	for (long i = 0; i < n; ++i) {
		for (long j = 0; j < n; ++j) {
			for (long k = 0; k < n; ++k) {
				res[i + j * n] += vals1[k + j * n] * vals2[i + k * n];
			}
		}
	}
	return res;
}

void EvaluatorUtils::squareMatSquareAndEqual(complex<double>* vals, long n) {
	long n2 = n * n;
	complex<double>* res = new complex<double>[n2];
	for (long i = 0; i < n; ++i) {
		for (long j = 0; j < n; ++j) {
			for (long k = 0; k < n; ++k) {
				res[i + j * n] += vals[k + j * n] * vals[i + k * n];
			}
		}
	}
	for (long i = 0; i < n2; ++i) {
		vals[i] = res[i];
	}
	delete[] res;
}
