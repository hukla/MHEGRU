#include "Plaintext.h"

Plaintext::Plaintext(long logp, long n0, long n1) : logp(logp), n0(n0), n1(n1) {
}


Plaintext::~Plaintext() {
	delete[] mx;
}



