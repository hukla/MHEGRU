#ifndef MHEAAN_STRINGUTILS_H_
#define MHEAAN_STRINGUTILS_H_

#include <NTL/ZZ.h>
#include <complex>
#include <iostream>

using namespace std;
using namespace NTL;

class StringUtils {
public:

	static void showVec(long* vals, long n);
	static void showVec(double* vals, long n);
	static void showVec(complex<double>* vals, long n);
	static void showVec(ZZ* vals, long n);

	static void showMat(long* vals, long nx, long ny);
	static void showMat(double* vals, long nx, long ny);
	static void showMat(complex<double>* vals, long nx, long ny);
	static void showMat(ZZ* vals, long nx, long ny);

	static void compare(complex<double> val1, complex<double> val2, string prefix);
	static void compare(complex<double>* vals1, complex<double>* vals2, long n, string prefix);
	static void compare(complex<double>* vals1, complex<double> val2, long n, string prefix);
	static void compare(complex<double> val1, complex<double>* vals2, long n, string prefix);

};

#endif
