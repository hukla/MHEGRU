#include "Ciphertext.h"

Ciphertext::Ciphertext(long logp, long logq, long n0, long n1) : logp(logp), logq(logq), n0(n0), n1(n1) {
}

Ciphertext::Ciphertext(const Ciphertext& o) : logp(o.logp), logq(o.logq), n0(o.n0), n1(o.n1) {
	for (long i = 0; i < N; ++i) {
		ax[i] = o.ax[i];
		bx[i] = o.bx[i];
	}
}

void Ciphertext::copyParams(const Ciphertext& o) {
	logp = o.logp;
	logq = o.logq;
	n0 = o.n0;
	n1 = o.n1;
}

void Ciphertext::copy(const Ciphertext& o) {
	copyParams(o);
	for (long i = 0; i < N; ++i) {
		ax[i] = o.ax[i];
		bx[i] = o.bx[i];
	}
}

void Ciphertext::free() {
	for (long i = 0; i < N; ++i) {
		clear(ax[i]);
		clear(bx[i]);
	}
}

Ciphertext::~Ciphertext() {
	delete[] ax;
	delete[] bx;
}
