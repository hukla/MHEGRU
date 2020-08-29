#include "StringUtils.h"


void StringUtils::showVec(long* vals, long n) {
	cout << "[";
	for (long i = 0; i < n; ++i) {
		cout << vals[i] << ", ";
	}
	cout << "]" << endl;
}

void StringUtils::showVec(double* vals, long n) {
	cout << "[";
	for (long i = 0; i < n; ++i) {
		cout << vals[i] << ", ";
	}
	cout << "]" << endl;
}

void StringUtils::showVec(complex<double>* vals, long n) {
	cout << "[";
	for (long i = 0; i < n; ++i) {
		cout << vals[i] << ", ";
	}
	cout << "]" << endl;
}


void StringUtils::showVec(ZZ* vals, long n) {
	cout << "[";
	for (long i = 0; i < n; ++i) {
		cout << vals[i] << ", ";
	}
	cout << "]" << endl;
}

void StringUtils::showMat(long* vals, long nx, long ny) {
	for (long iy = 0; iy < ny; ++iy) {
		cout << "[";
		for (long ix = 0; ix < nx; ++ix) {
			cout << vals[ix + iy * nx] << ", ";
		}
		cout << "]" << endl;
	}
}

void StringUtils::showMat(double* vals, long nx, long ny) {
	for (long iy = 0; iy < ny; ++iy) {
		cout << "[";
		for (long ix = 0; ix < nx; ++ix) {
			cout << vals[ix + iy * nx] << ", ";
		}
		cout << "]" << endl;
	}
}

void StringUtils::showMat(complex<double>* vals, long nx, long ny) {
	for (long iy = 0; iy < ny; ++iy) {
		cout << "[";
		for (long ix = 0; ix < nx; ++ix) {
			cout << vals[ix + iy * nx] << ", ";
		}
		cout << "]" << endl;
	}
}

void StringUtils::showMat(ZZ* vals, long nx, long ny) {
	for (long iy = 0; iy < ny; ++iy) {
		cout << "[";
		for (long ix = 0; ix < nx; ++ix) {
			cout << vals[ix + iy * nx] << ", ";
		}
		cout << "]" << endl;
	}
}



void StringUtils::compare(complex<double> val1, complex<double> val2, string prefix) {
	cout << "---------------------" << endl;
	cout << "m" + prefix + ":" << val1 << endl;
	cout << "d" + prefix + ":" << val2 << endl;
	cout << "e" + prefix + ":" << val1-val2 << endl;
	cout << "---------------------" << endl;
}

void StringUtils::compare(complex<double>* vals1, complex<double>* vals2, long n, string prefix) {
	for (long i = 0; i < n; ++i) {
		cout << "---------------------" << endl;
		cout << "m" + prefix + ": " << i << " :" << vals1[i] << endl;
		cout << "d" + prefix + ": " << i << " :" << vals2[i] << endl;
		cout << "e" + prefix + ": " << i << " :" << vals1[i]-vals2[i] << endl;
		cout << "---------------------" << endl;
	}
}

void StringUtils::compare(complex<double>* vals1, complex<double> val2, long n, string prefix) {
	for (long i = 0; i < n; ++i) {
		cout << "---------------------" << endl;
		cout << "m" + prefix + ": " << i << " :" << vals1[i] << endl;
		cout << "d" + prefix + ": " << i << " :" << val2 << endl;
		cout << "e" + prefix + ": " << i << " :" << vals1[i]-val2 << endl;
		cout << "---------------------" << endl;
	}
}

void StringUtils::compare(complex<double> val1, complex<double>* vals2, long n, string prefix) {
	for (long i = 0; i < n; ++i) {
		cout << "---------------------" << endl;
		cout << "m" + prefix + ": " << i << " :" << val1 << endl;
		cout << "d" + prefix + ": " << i << " :" << vals2[i] << endl;
		cout << "e" + prefix + ": " << i << " :" << val1-vals2[i] << endl;
		cout << "---------------------" << endl;
	}
}
