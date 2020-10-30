#include "RNN.h"dd

#include <NTL/ZZ.h>

#include <math.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <array>
#include <random>
#include <chrono>

using namespace NTL;

void RNN::readV(double* v, string path) {
	ifstream openFile(path.data());
	if(openFile.is_open()) {
		string line, temp;
		size_t start, end;
		long i = 0;
		while(getline(openFile, line)) {
			v[i++] = atof(line.c_str());
		}
	} else {
		cout << "Error: cannot read file" << endl;
	}
}

void RNN::readVt(double* v, string path) {
	ifstream openFile(path.data());
	if(openFile.is_open()) {
		string line, temp;
		size_t start, end;
		long j = 0;
		getline(openFile, line);
		do {
			end = line.find_first_of(',', start);
			temp = line.substr(start,end);
			v[j] = atof(temp.c_str());
			start = end + 1;
			j++;
		} while(start);
	} else {
		cout << "Error: cannot read file:" << path << endl;
	}
}

void RNN::readM(double** M, string path) {
	ifstream openFile(path.data());
	if(openFile.is_open()) {
		string line, temp;
		size_t start=0, end;
		long i = 0;
		long j;
		while(getline(openFile, line)) {
			j = 0;
			do {
				end = line.find_first_of(',', start);
				temp = line.substr(start,end);
				M[i][j] = atof(temp.c_str());
				start = end + 1;
				j++;
			} while(start);
			i++;
		}
	} else {
		cout << "Error: cannot read file " << path << endl;
	}
}



void RNN::evalMV(double* MV, double** M, double* V, long Mrow, long Mcol) {
	for (int row = 0; row < Mrow; ++row) {
		MV[row] = 0.;
		for (int col = 0; col < Mcol; ++col) {
			MV[row] += (M[row][col] * V[col]);
		}
	}
}

void RNN::evalMVx(double* MV, double** M, double* V, long Mrow, long Mcol) {
	for (int row = 0; row < Mrow; ++row) {
		MV[row] = 0.;
		for (int col = 0; col < Mcol; ++col) {
			MV[row] += (M[col][row] * V[col]);
		}
	}
}

void RNN::evalAdd(double* vr, double* v1, double* v2, long n) {
	for (int i = 0; i < n; ++i) {
		vr[i] = v1[i] + v2[i];
	}
}

void RNN::evalAddAndEqual(double* v1, double* v2, long n) {
	for (int i = 0; i < n; ++i) {
		v1[i] += v2[i];
	}
}

void RNN::evalMul(double* vr, double* v1, double* v2, long n) {
	for (int i = 0; i < n; ++i) {
		vr[i] = v1[i] * v2[i];
	}
}

void RNN::evalMulAndEqual(double* v1, double* v2, long n) {
	for (int i = 0; i < n; ++i) {
		v1[i] *= v2[i];
	}
}

void RNN::evalOnem(double* vr, double* v, long n) {
	for (int i = 0; i < n; ++i) {
		vr[i] = 1. - v[i];
	}
}

void RNN::evalSigmoid(double* x, long n) {
	double sigmoid3[3] = {0.5,0.1424534,-0.0013186}; // x in [-6, 6]
	for (long i = 0; i < n; ++i) {
		x[i] = sigmoid3[2] * pow(x[i], 3) + sigmoid3[1] * x[i] + sigmoid3[0];
	}

}

void RNN::evalSigmoid(double* x, long n, int order=7) {
	double sigmoid3[3] = {0.5,0.1424534,-0.0013186}; // x in [-6, 6]
	double sigmoid5[4] = {0.5, 0.19130488174364327, -0.004596051850950526, 4.223017442715702e-05}; // x in [-8, 8]
	double sigmoid7[5] = {0.5, 0.21689567455156572, -0.008194757398825834, 0.00016593568955483007, -1.1965564496759948e-06}; // x in [-8, 8]

	if (order == 3) {
		for (long i = 0; i < n; ++i) {
			x[i] = sigmoid3[2] * pow(x[i], 3) + sigmoid3[1] * x[i] + sigmoid3[0];
		}
	} else if (order == 5) {
		for (long i = 0; i < n; ++i) {
			x[i] = sigmoid5[3] * pow(x[i], 5) + sigmoid5[2] * pow(x[i], 3) + sigmoid5[1] * x[i] + sigmoid5[0];
		}
	} else {
		for (long i = 0; i < n; ++i) {
			x[i] = sigmoid7[4] * pow(x[i], 7) + sigmoid7[3] * pow(x[i], 5) + sigmoid7[2] * pow(x[i], 3) + sigmoid7[1] * x[i] + sigmoid7[0];
		}
	}
}


void RNN::evalTanh(double* x, long n) {
	double tanh5[4] = {0.00038260975296624476, 0.7652194684902834, -0.07353682621097166, 0.002702731463794033};  // RMSE: 0.0543
	for (long i = 0; i < n; ++i) {
		x[i] = tanh5[3] * pow(x[i], 5) +
				tanh5[2] * pow(x[i], 3) + tanh5[1] * x[i];
//		x[i] = (x[i] > 15.0) ? 1.0 : (x[i] < -15.0) ? -1.0 : tanh(x[i]);
	}
}

void RNN::evalTanh(double* x, long n, int order=7) {
	double tanh5[4] = {0.00038260975296624476, 0.7652194684902834, -0.07353682621097166, 0.002702731463794033};  // RMSE: 0.0543
	double tanh7[5] = {0.00043379132689107155, 0.8675825874601593, -0.13111610042441557, 0.010619884719547454, -0.0003063185603617004};  // RMSE: 0.0252

	if (order == 5) {
		for (long i = 0; i < n; ++i) {
			x[i] = tanh5[3] * pow(x[i], 5) + tanh5[2] * pow(x[i], 3) + tanh5[1] * x[i] + tanh5[0];
		}
	} else {
		for (long i = 0; i < n; ++i) {
			x[i] = tanh7[4] * pow(x[i], 7) + tanh7[3] * pow(x[i], 5) + tanh7[2] * pow(x[i], 3) + tanh7[1] * x[i] + tanh7[0];
		}
	}
}

void RNN::printv(double* v, string name, long n) {
	cout << "-----------" << name << "--------------" << endl;
	double mm = 0.0;
	for (int i = 0; i < n; ++i) {
		cout << v[i] << ",";
		mm = max(mm, abs(v[i]));
	}
	cout << endl;
	cout << "max: " << mm << endl;
	cout << "-------------------------------------" << endl;
}

void RNN::printM(double** M, string name, long nrow, long ncol) {
	cout << "-----------" << name << "--------------" << endl;
	for (int j = 0; j < ncol; ++j) {
		for (int i = 0; i < nrow; ++i) {
			cout << M[i][j] << ",";
		}
		cout << endl;
	}
	cout << "-------------------------------------" << endl;

}
