#ifndef MPHEAAN_PLAINTEXT_H_
#define MPHEAAN_PLAINTEXT_H_

#include "Params.h"
#include <NTL/ZZ.h>

using namespace std;
using namespace NTL;

class Plaintext {
public:

	ZZ* mx = new ZZ[N];

	long logp;

	long n0;
	long n1;

	Plaintext(long logp = 0, long n0 = 0, long n1 = 0);

	~Plaintext();

};

#endif
