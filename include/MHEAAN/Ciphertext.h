#ifndef MPHEAAN_CIPHERTEXT_H_
#define MPHEAAN_CIPHERTEXT_H_

#include <NTL/ZZ.h>
#include "Params.h"

using namespace std;
using namespace NTL;

class Ciphertext {
public:

	ZZ* ax = new ZZ[N];
	ZZ* bx = new ZZ[N];

	long logp;
	long logq;

	long n0;
	long n1;

	Ciphertext(long logp = 0, long logq = 0, long n0 = 0, long n1 = 0);

	Ciphertext(const Ciphertext& o);

	void copyParams(const Ciphertext& o);

	void copy(const Ciphertext& o);

	void free();

	virtual ~Ciphertext();
};

#endif
