#ifndef MPHEAAN_EVALUATORUTILS_H_
#define MPHEAAN_EVALUATORUTILS_H_

#include <NTL/RR.h>
#include <NTL/ZZ.h>
#include <complex>

using namespace std;
using namespace NTL;

class EvaluatorUtils {
public:


	//----------------------------------------------------------------------------------
	//   RANDOM REAL AND COMPLEX NUMBERS
	//----------------------------------------------------------------------------------


	static double randomReal(double bound = 1.0);

	static double randomRealSigned(double bound = 1.0);

	static complex<double> randomComplex(double bound = 1.0);

	static complex<double> randomComplexSigned(double bound = 1.0);

	static complex<double> randomCircle(double anglebound = 1.0);

	static double* randomRealArray(long n, double bound = 1.0);

	static double* randomRealSignedArray(long n, double bound = 1.0);

	static complex<double>* randomComplexArray(long n, double bound = 1.0);

	static complex<double>* randomComplexSignedArray(long n, double bound = 1.0);

	static complex<double>* randomCircleArray(long n, double bound = 1.0);


	//----------------------------------------------------------------------------------
	//   DOUBLE & RR <-> ZZ
	//----------------------------------------------------------------------------------


	static double scaleDownToReal(ZZ& x, long logp);

	static ZZ scaleUpToZZ(double x, long logp);

	static ZZ scaleUpToZZ(RR& x, long logp);


	//----------------------------------------------------------------------------------
	//   ROTATIONS
	//----------------------------------------------------------------------------------


	static void leftRotateAndEqual(complex<double>* vals, long n0, long n1, long r0, long r1);

	static void rightRotateAndEqual(complex<double>* vals, long n0, long n1, long r0, long r1);


	//----------------------------------------------------------------------------------
	//   MATRIX
	//----------------------------------------------------------------------------------

	static complex<double>* transpose(complex<double>* vals, long n);

	static complex<double>* squareMatMult(complex<double>* vals1, complex<double>* vals2, long n);

	static void squareMatSquareAndEqual(complex<double>* vals, long n);
};

#endif
