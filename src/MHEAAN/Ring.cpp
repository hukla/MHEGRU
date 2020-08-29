#include "Ring.h"

#include <NTL/BasicThreadPool.h>
#include "EvaluatorUtils.h"
#include "StringUtils.h"


Ring::Ring() {

	uint64_t g0 = 5;
	uint64_t g0Pow = 1;
	for (long i = 0; i < N0h; ++i) {
		gM0Pows[i] = g0Pow;
		g0Pow *= g0;
		g0Pow %= M0;
	}
	gM0Pows[N0h] = gM0Pows[0];

	uint64_t g1 = multiplier.findPrimitiveRoot(M1);
	uint64_t g1Pow = 1;
	for (long i = 0; i < N1; ++i) {
		gM1Pows[i] = g1Pow;
		g1Pow *= g1;
		g1Pow %= M1;
	}
	gM1Pows[N1] = gM1Pows[0];

	for (long j = 0; j < M0; ++j) {
		double angle = 2.0 * M_PI * j / M0;
		ksiM0Pows[j].real(cos(angle));
		ksiM0Pows[j].imag(sin(angle));
	}
	ksiM0Pows[M0] = ksiM0Pows[0];

	for (long j = 0; j < N1; ++j) {
		double angle = 2.0 * M_PI * j / N1;
		ksiN1Pows[j].real(cos(angle));
		ksiN1Pows[j].imag(sin(angle));
	}
	ksiN1Pows[N1] = ksiN1Pows[0];

	for (long j = 0; j < M1; ++j) {
		double angle = 2.0 * M_PI * j / M1;
		ksiM1Pows[j].real(cos(angle));
		ksiM1Pows[j].imag(sin(angle));
	}
	ksiM1Pows[M1] = ksiM1Pows[0];

	for (long logn1 = 0; logn1 < logN1 + 1; ++logn1) {
		long n1 = 1 << logn1;
		long gap = 1 << (logN1 - logn1);
		dftM1Pows[logn1] = new complex<double>[n1+1];
		dftM1NTTPows[logn1] = new complex<double>[n1+1];
		for (long i = 0; i < n1; ++i) {
			for (long k = 0; k < gap; ++k) {
				dftM1Pows[logn1][i] += ksiM1Pows[gM1Pows[i + k * n1]];
			}
			dftM1NTTPows[logn1][i] = dftM1Pows[logn1][i];
		}
		DFTX1(dftM1NTTPows[logn1], n1);

		dftM1Pows[logn1][n1] = dftM1Pows[logn1][0];
		dftM1NTTPows[logn1][n1] = dftM1NTTPows[logn1][0];
	}

	qvec[0] = ZZ(1);
	for (long i = 1; i < logQQ + 1; ++i) {
		qvec[i] = qvec[i - 1] << 1;
	}
}

void Ring::addBootContext(long logn0, long logn1, long logp) {
	if (bootContextMap.find({logn0, logn1}) == bootContextMap.end()) {
		long n0 = 1 << logn0;
		long logk0 = logn0 >> 1;
		long k0 = 1 << logk0;

		uint64_t** rpVec = new uint64_t*[n0];
		uint64_t** rpInvVec = new uint64_t*[n0];
		uint64_t* rp1 = NULL;
		uint64_t* rp2 = NULL;

		long* bndVec = new long[n0];
		long* bndInvVec = new long[n0];
		long bnd1 = 0;
		long bnd2 = 0;

		long np;
		complex<double>* pvals = new complex<double>[n0];
		ZZ* pVec = new ZZ[N0];

		long gap0 = N0h >> logn0;
		long deg;
		for (long ki = 0; ki < n0; ki += k0) {
			for (long pos = ki; pos < ki + k0; ++pos) {
				for (long i = 0; i < n0 - pos; ++i) {
					deg = ((M0 - gM0Pows[i + pos]) * i * gap0) % M0;
					pvals[i] = ksiM0Pows[deg];
				}
				for (long i = n0 - pos; i < n0; ++i) {
					deg = ((M0 - gM0Pows[i + pos - n0]) * i * gap0) % M0;
					pvals[i] = ksiM0Pows[deg];
				}
				EvaluatorUtils::rightRotateAndEqual(pvals, n0, 1, ki, 0);
				IEMBX0(pvals, n0);
				for (long i = 0, jd = N0h, id = 0; i < n0; ++i, jd += gap0, id += gap0) {
					pVec[id] = EvaluatorUtils::scaleUpToZZ(pvals[i].real(), logp);
					pVec[jd] = EvaluatorUtils::scaleUpToZZ(pvals[i].imag(), logp);
				}
				bndVec[pos] = MaxBits(pVec, N0);
				np = ceil((logQ + bndVec[pos] + logN0 + 3)/(double)pbnd);
				rpVec[pos] = new uint64_t[np << logN0];
				toNTTX0(rpVec[pos], pVec, np);
			}
		}

		for (long ki = 0; ki < n0; ki += k0) {
			for (long pos = ki; pos < ki + k0; ++pos) {
				for (long i = 0; i < n0 - pos; ++i) {
					deg = (gM0Pows[i] * (i + pos) * gap0) % M0;
					pvals[i] = ksiM0Pows[deg];
				}
				for (long i = n0 - pos; i < n0; ++i) {
					deg = (gM0Pows[i] * (i + pos - n0) * gap0) % M0;
					pvals[i] = ksiM0Pows[deg];
				}
				EvaluatorUtils::rightRotateAndEqual(pvals, n0, 1, ki, 0);
				IEMBX0(pvals, n0);
				for (long i = 0, jd = N0h, id = 0; i < n0; ++i, jd += gap0, id += gap0) {
					pVec[id] = EvaluatorUtils::scaleUpToZZ(pvals[i].real(), logp);
					pVec[jd] = EvaluatorUtils::scaleUpToZZ(pvals[i].imag(), logp);
				}
				bndInvVec[pos] = MaxBits(pVec, N0);
				np = ceil((logQ + bndInvVec[pos] + logN0 + 3)/(double)pbnd);
				rpInvVec[pos] = new uint64_t[np << logN0];
				toNTTX0(rpInvVec[pos], pVec, np);
			}
		}

		delete[] pvals;
		delete[] pVec;

		BootContext* bootContext = new BootContext(rpVec, rpInvVec, rp1, rp2, bndVec, bndInvVec, bnd1, bnd2, logp);
		bootContextMap.insert(pair<pair<long, long>, BootContext&>({logn0, logn1}, *bootContext));
	}
}

void Ring::addSqrMatContext(long logn, long logp) {
	if (sqrMatContextMap.find(logn) == sqrMatContextMap.end()) {
		long n = (1 << logn);

		ZZ** mvec = new ZZ*[n];
		double* tmp = new double[n * n]();
		for (long i = 0; i < n; ++i) {
			for (long j = 0; j < n; ++j) {
				tmp[j + (((j + n - i) % n) * n)] = 1.0;
			}

			mvec[i] = new ZZ[N];
			encode(mvec[i], tmp, n, n, logp);

			for (long j = 0; j < n; ++j) {
				tmp[j + (((j + n - i) % n) * n)] = 0.0;
			}
		}
		delete[] tmp;

		SqrMatContext* sqrMatContext = new SqrMatContext(mvec, logp);
		sqrMatContextMap.insert(pair<long, SqrMatContext&>(logn, *sqrMatContext));
	}
}

void Ring::arrayBitReverse(complex<double>* vals, long n) {
	for (long i = 1, j = 0; i < n; ++i) {
		long bit = n >> 1;
		for (; j >= bit; bit>>=1) {
			j -= bit;
			if (j == 0 && bit == 0) break; // TODO
		}
		j += bit;
		if(i < j) {
			swap(vals[i], vals[j]);
		}
	}
}

void Ring::DFTX1(complex<double>* vals, long n1) {
	arrayBitReverse(vals, n1);
	complex<double> u, v;
	for (long len = 2; len <= n1; len <<= 1) {
		long lenh = len >> 1;
		long gap = N1 / len;
		for (long i = 0; i < n1; i += len) {
			for (long j = 0; j < lenh; ++j) {
				long idx = j * gap;
				u = vals[i + j];
				v = vals[i + j + lenh];
				v *= ksiN1Pows[idx];
				vals[i + j] = u + v;
				vals[i + j + lenh] = u - v;
			}
		}
	}
}

void Ring::IDFTX1(complex<double>* vals, long n1) {
	arrayBitReverse(vals, n1);
	for (long len = 2; len <= n1; len <<= 1) {
		long lenh = len >> 1;
		long gap = N1 / len;
		for (long i = 0; i < n1; i += len) {
			for (long j = 0; j < lenh; ++j) {
				long idx = N1 - (j * gap);
				complex<double> u = vals[i + j];
				complex<double> v = vals[i + j + lenh];
				v *= ksiN1Pows[idx];
				vals[i + j] = u + v;
				vals[i + j + lenh] = u - v;
			}
		}
	}
	for (long i = 0; i < n1; ++i) {
		vals[i] /= n1;
	}
}

void Ring::EMBX0(complex<double>* vals, long n0) {
	arrayBitReverse(vals, n0);
	for (long len = 2; len <= n0; len <<= 1) {
		for (long i = 0; i < n0; i += len) {
			long lenh = len >> 1;
			long lenq = len << 2;
			long gap = M0 / lenq;
			for (long j = 0; j < lenh; ++j) {
				long idx = (gM0Pows[j] % lenq) * gap;
				complex<double> u = vals[i + j];
				complex<double> v = vals[i + j + lenh];
				v *= ksiM0Pows[idx];
				vals[i + j] = u + v;
				vals[i + j + lenh] = u - v;
			}
		}
	}
}

void Ring::IEMBX0(complex<double>* vals, long n0) {
    complex<double> u;
	complex<double> v;

	for (long len = n0; len >= 1; len >>= 1) {
		for (long i = 0; i < n0; i += len) {
			long lenh = len >> 1;
			long lenq = len << 2;
			long gap = M0 / lenq;
			for (long j = 0; j < lenh; ++j) {
				long idx = (lenq - (gM0Pows[j] % lenq)) * gap;
				u = vals[i + j] + vals[i + j + lenh];
				v = vals[i + j] - vals[i + j + lenh];
				v *= ksiM0Pows[idx];
				vals[i + j] = u;
				vals[i + j + lenh] = v;
			}
		}
	}
	arrayBitReverse(vals, n0);
	for (long i = 0; i < n0; ++i) {
		vals[i] /= n0;
	}
}

void Ring::EMBX1(complex<double>* vals, long n1) {
	long logn1 = (long)log2(n1);
	DFTX1(vals, n1);
	for (long i = 0; i < n1; ++i) {
		vals[i] *= dftM1NTTPows[logn1][i];
	}
	IDFTX1(vals, n1);
}

void Ring::IEMBX1(complex<double>* vals, long n1) {
	long logn1 = (long)log2(n1);
	DFTX1(vals, n1);
	for (long i = 0; i < n1; ++i) {
		vals[i] /= dftM1NTTPows[logn1][i];
	}
	IDFTX1(vals, n1);
}

void Ring::EMB(complex<double>* vals, long n0, long n1) {
	complex<double>* tmp = new complex<double>[n1]();
	for (long i = 0; i < n1; ++i) {
		EMBX0(vals + (i * n0), n0);
	}
	for (long i = 0; i < n0; ++i) {
		for (long j = 0; j < n1; ++j) {
			tmp[j] = vals[i + (j * n0)];
		}
		EMBX1(tmp, n1);
		for (long j = 0; j < n1; ++j) {
			vals[i + (j * n0)] = tmp[j];
		}
	}
	delete[] tmp;
}

void printVals(complex<double>* vals, long n0, long n1) {
	for (long y = 0; y < n1; y++) {
		for (long x = 0; x < n0; x++) {
			cout << vals[x + n0 * y];
		}
		cout << endl;
	}
}

void Ring::IEMB(complex<double>* vals, long n0, long n1) {
	complex<double>* tmp = new complex<double>[n1]();

	for (long i1 = 0; i1 < n1; ++i1) {
		IEMBX0(vals + (i1 * n0), n0);
	}
	for (long i0 = 0; i0 < n0; ++i0) {
		for (long i1 = 0; i1 < n1; ++i1) {
			tmp[i1] = vals[i0 + (i1 * n0)];
		}
		IEMBX1(tmp, n1);
		for (long i1 = 0; i1 < n1; ++i1) {
			vals[i0 + (i1 * n0)] = tmp[i1];
		}
	}

	delete[] tmp;
}

void Ring::encode(ZZ* mx, complex<double>* vals, long n0, long n1, long logp) {
	long gap0 = N0h / n0;
	long gap1 = N1 / n1;

	complex<double>* uvals = new complex<double>[n0 * n1];
	for (long i = 0; i < n0 * n1; ++i) {
		uvals[i] = vals[i];
	}

	IEMB(uvals, n0, n1);
	for (long i = 0, ii = N0h, ir = 0; i < n0; ++i, ii += gap0, ir += gap0) {
		for (long j = 0; j < n1; ++j) {
			for (long g = 0; g < gap1; ++g) {
				mx[ir + N0 * (j + g * n1)] = EvaluatorUtils::scaleUpToZZ(uvals[i + n0 * j].real(), logp);
				mx[ii + N0 * (j + g * n1)] = EvaluatorUtils::scaleUpToZZ(uvals[i + n0 * j].imag(), logp);
			}
		}
	}
	delete[] uvals;
}

void Ring::encode(ZZ* mx, double* vals, long n0, long n1, long logp) {
	long gap0 = N0h / n0;
	long gap1 = N1 / n1;

	complex<double>* uvals = new complex<double>[n0 * n1];
	for (long i = 0; i < n0 * n1; ++i) {
		uvals[i].real(vals[i]);
	}

	IEMB(uvals, n0, n1);
	for (long i0 = 0, ii0 = N0h, ir0 = 0; i0 < n0; ++i0, ii0 += gap0, ir0 += gap0) {
		for (long i1 = 0; i1 < n1; ++i1) {
			for (long g1 = 0; g1 < gap1; ++g1) {
				mx[ir0 + N0 * (i1 + g1 * n1)] = EvaluatorUtils::scaleUpToZZ(uvals[i0 + n0 * i1].real(), logp);
				mx[ii0 + N0 * (i1 + g1 * n1)] = EvaluatorUtils::scaleUpToZZ(uvals[i0 + n0 * i1].imag(), logp);
			}
		}
	}
	delete[] uvals;
}

void Ring::decode(complex<double>* vals, ZZ* mx, long n0, long n1, long logp) {
long gap0 = N0h / n0;
	long gap1 = N1 / n1;

	for (long i0 = 0, ii0 = N0h, ir0 = 0; i0 < n0; ++i0, ii0 += gap0, ir0 += gap0) {
		for (long i1 = 0; i1 < n1; ++i1) {
			for (long g1 = 0; g1 < gap1; ++g1) {
				vals[i0 + n0 * i1].real(EvaluatorUtils::scaleDownToReal(mx[ir0 + N0 * (i1 + g1 * n1)], logp));
				vals[i0 + n0 * i1].imag(EvaluatorUtils::scaleDownToReal(mx[ii0 + N0 * (i1 + g1 * n1)], logp));
			}
		}
	}

	EMB(vals, n0, n1);
}


//----------------------------------------------------------------------------------
//   MULTIPLICATION
//----------------------------------------------------------------------------------


long Ring::MaxBits(ZZ* f, long n) {
	long i, m;
	m = 0;
	for (i = 0; i < n; i++) {
		m = max(m, NumBits(f[i]));
	}
	return m;
}

void Ring::toNTTX0(uint64_t* ra, ZZ* a, long np) {
	multiplier.toNTTX0(ra, a, np);
}

void Ring::toNTTX1(uint64_t* ra, ZZ* a, long np) {
	multiplier.toNTTX1(ra, a, np);
}

void Ring::toNTT(uint64_t* ra, ZZ* a, long np) {
	multiplier.toNTT(ra, a, np);
}

void Ring::addNTTAndEqual(uint64_t* ra, uint64_t* rb, long np) {
	multiplier.addNTTAndEqual(ra, rb, np);
}

void Ring::multX0(ZZ* x, ZZ* a, ZZ* b, long np, const ZZ& q) {
	multiplier.multX0(x, a, b, np, q);
}

void Ring::multX0AndEqual(ZZ* a, ZZ* b, long np, const ZZ& q) {
	multiplier.multX0AndEqual(a, b, np, q);
}

void Ring::multNTTX0(ZZ* x, ZZ* a, uint64_t* rb, long np, const ZZ& q) {
	multiplier.multNTTX0(x, a, rb, np, q);
}

void Ring::multNTTX0AndEqual(ZZ* a, uint64_t* rb, long np, const ZZ& q) {
	multiplier.multNTTX0AndEqual(a, rb, np, q);
}

void Ring::multDNTTX0(ZZ* x, uint64_t* ra, uint64_t* rb, long np, const ZZ& q) {
	multiplier.multDNTTX0(x, ra, rb, np, q);
}

void Ring::multX1(ZZ* x, ZZ* a, ZZ* b, long np, const ZZ& q) {
	multiplier.multX1(x, a, b, np, q);
}

void Ring::multX1AndEqual(ZZ* a, ZZ* b, long np, const ZZ& q) {
	multiplier.multX1AndEqual(a, b, np, q);
}

void Ring::multNTTX1(ZZ* x, ZZ* a, uint64_t* b, long np, const ZZ& q) {
	multiplier.multNTTX1(x, a, b, np, q);
}

void Ring::multNTTX1AndEqual(ZZ* a, uint64_t* b, long np, const ZZ& q) {
	multiplier.multNTTX1AndEqual(a, b, np, q);
}

void Ring::mult(ZZ* x, ZZ* a, ZZ* b, long np, const ZZ& q) {
	multiplier.mult(x, a, b, np, q);
}

void Ring::multAndEqual(ZZ* a, ZZ* b, long np, const ZZ& q) {
	multiplier.multAndEqual(a, b, np, q);
}

void Ring::multNTT(ZZ* x, ZZ* a, uint64_t* rb, long np, const ZZ& q) {
	multiplier.multNTT(x, a, rb, np, q);
}

void Ring::multNTTAndEqual(ZZ* a, uint64_t* rb, long np, const ZZ& q) {
	multiplier.multNTTAndEqual(a, rb, np, q);
}

void Ring::multDNTT(ZZ* x, uint64_t* ra, uint64_t* rb, long np, const ZZ& q) {
	multiplier.multDNTT(x, ra, rb, np, q);
}

void Ring::square(ZZ* x, ZZ* a, long np, const ZZ& q) {
	multiplier.square(x, a, np, q);
}

void Ring::squareAndEqual(ZZ* a, long np, const ZZ& q) {
	multiplier.squareAndEqual(a, np, q);
}

void Ring::squareNTT(ZZ* x, uint64_t* ra, long np, const ZZ& q) {
	multiplier.squareNTT(x, ra, np, q);
}


//----------------------------------------------------------------------------------
//   OTHER
//----------------------------------------------------------------------------------

void Ring::normalize(ZZ* res, ZZ* p, const ZZ& q) {
	ZZ qh = q/2;
	for (int i = 0; i < N; ++i) {
		if(p[i] > qh) res[i] = p[i] - q;
		else if(p[i] < -qh) res[i] = p[i] + q;
		else res[i] = p[i];
	}
}

void Ring::normalizeAndEqual(ZZ* p, const ZZ& q) {
	ZZ qh = q/2;
	for (int i = 0; i < N; ++i) {
		if(p[i] > qh) p[i] -= q;
		else if(p[i] < -qh) p[i] += q;
	}
}

void Ring::mod(ZZ* res, ZZ* p, const ZZ& q) {
	for (long i = 0; i < N; ++i) {
		rem(res[i], p[i], q);
	}
}

void Ring::modAndEqual(ZZ* p, const ZZ& q) {
	for (long i = 0; i < N; ++i) {
		rem(p[i], p[i], q);
	}
}

void Ring::negate(ZZ* res, ZZ* p) {
	for (long i = 0; i < N; ++i) {
		res[i] = -p[i];
	}
}

void Ring::negateAndEqual(ZZ* p) {
	for (long i = 0; i < N; ++i) {
		p[i] = -p[i];
	}
}

void Ring::add(ZZ* res, ZZ* p1, ZZ* p2, const ZZ& q) {
	for (long i = 0; i < N; ++i) {
		AddMod(res[i], p1[i], p2[i], q);
	}
}

void Ring::addAndEqual(ZZ* p1, ZZ* p2, const ZZ& q) {
	for (long i = 0; i < N; ++i) {
		AddMod(p1[i], p1[i], p2[i], q);
	}
}

void Ring::sub(ZZ* res, ZZ* p1, ZZ* p2, const ZZ& q) {
	for (long i = 0; i < N; ++i) {
		AddMod(res[i], p1[i], -p2[i], q);
	}
}

void Ring::subAndEqual(ZZ* p1, ZZ* p2, const ZZ& q) {
	for (long i = 0; i < N; ++i) {
		AddMod(p1[i], p1[i], -p2[i], q);
	}
}

void Ring::subAndEqual2(ZZ* p1, ZZ* p2, const ZZ& q) {
	for (long i = 0; i < N; ++i) {
		AddMod(p2[i], p1[i], -p2[i], q);
	}
}

void Ring::multByMonomial(ZZ* res, ZZ* p, long deg0, long deg1, const ZZ& q) {
	for (long i = 0; i < N0; ++i) {
		for (long j = 1; j < M1; ++j) {
			long resdeg0 = (deg0 + i) % M0;
			long resdeg1 = (deg1 + j) % M1;

			if(resdeg0 < N0) {
				if(resdeg1 > 0) {
					AddMod(res[resdeg0 + ((resdeg1-1) << logN0)], res[resdeg0 + ((resdeg1-1) << logN0)], p[i + ((j-1) << logN0)], q);
				} else {
					for (long k = 0; k < N1; ++k) {
						AddMod(res[resdeg0 + (k << logN0)], res[resdeg0 + (k << logN0)], -p[i + ((j-1) << logN0)], q);
					}
				}
			} else {
				if(resdeg1 > 0) {
					AddMod(res[(resdeg0 - N0) + ((resdeg1-1) << logN0)], res[(resdeg0 - N0) + ((resdeg1-1) << logN0)], -p[i + ((j-1) << logN0)], q);
				} else {
					for (long k = 0; k < N1; ++k) {
						AddMod(res[(resdeg0 - N0) + (k << logN0)], res[(resdeg0 - N0) + (k << logN0)], p[i + ((j-1) << logN0)], q);
					}
				}
			}
		}
	}
}

void Ring::multByMonomialAndEqual(ZZ* p, long deg0, long deg1, const ZZ& q) {
	ZZ res[N];
	for (long i = 0; i < N0; ++i) {
		for (long j = 1; j < M1; ++j) {
			long resdeg0 = (deg0 + i) % M0;
			long resdeg1 = (deg1 + j) % M1;

			if(resdeg0 < N0) {
				if(resdeg1 > 0) {
					AddMod(res[resdeg0 + ((resdeg1-1) << logN0)], res[resdeg0 + ((resdeg1-1) << logN0)], p[i + ((j-1) << logN0)], q);
				} else {
					for (long k = 0; k < N1; ++k) {
						AddMod(res[resdeg0 + (k << logN0)], res[resdeg0 + (k << logN0)], -p[i + ((j-1) << logN0)], q);
					}
				}
			} else {
				if(resdeg1 > 0) {
					AddMod(res[(resdeg0 - N0) + ((resdeg1-1) << logN0)], res[(resdeg0 - N0) + ((resdeg1-1) << logN0)], -p[i + ((j-1) << logN0)], q);
				} else {
					for (long k = 0; k < N1; ++k) {
						AddMod(res[(resdeg0 - N0) + (k << logN0)], res[(resdeg0 - N0) + (k << logN0)], p[i + ((j-1) << logN0)], q);
					}
				}
			}
		}
	}

	for (long i = 0; i < N; ++i) {
		p[i] = res[i];
	}
}

void Ring::multByConst(ZZ* res, ZZ* p, ZZ& cnst, const ZZ& q) {
	for (long i = 0; i < N; ++i) {
		MulMod(res[i], p[i], cnst, q);
	}
}

void Ring::multByConstAndEqual(ZZ* p, ZZ& cnst, const ZZ& q) {
	for (long i = 0; i < N; ++i) {
		MulMod(p[i], p[i], cnst, q);
	}
}


//----------------------------------------------------------------------------------
//   SHIFTING
//----------------------------------------------------------------------------------


void Ring::leftShift(ZZ* res, ZZ* p, long bits, const ZZ& q) {
	for (long i = 0; i < N; ++i) {
		res[i] = p[i] << bits;
		res[i] %= q;
	}
}

void Ring::leftShiftAndEqual(ZZ* p, long bits, const ZZ& q) {
	for (long i = 0; i < N; ++i) {
		p[i] <<= bits;
		p[i] %= q;
	}
}

void Ring::rightShift(ZZ* res, ZZ* p, long bits) {
	for (long i = 0; i < N; ++i) {
		res[i] = p[i] >> bits;
	}
}

void Ring::rightShiftAndEqual(ZZ* p, long bits) {
	for (long i = 0; i < N; ++i) {
		p[i] >>= bits;
	}
}


//----------------------------------------------------------------------------------
//   ROTATION & CONJUGATION & TRANSPOSITION
//----------------------------------------------------------------------------------


void Ring::leftRotate(ZZ* res, ZZ* p, long r0, long r1) {
	long deg0 = gM0Pows[r0];
	for (long j = 0; j < N; j += N0) {
		for (long i = 0; i < N0; ++i) {
			long ipow = i * deg0;
			long shift = ipow % M0;
			if (shift < N0) {
				res[shift + j] = p[i + j];
			} else {
				res[shift - N0 + j] = -p[i + j];
			}
		}
	}

	r1 %= N1;
	if(r1 != 0) {
		long divisor = GCD(r1, N1);
		long steps = N1 / divisor;
		for (long i = 0; i < N0; ++i) {
			for (long j = 0; j < divisor; ++j) {
				ZZ tmp = res[i + (j << logN0)];
				long idx = j;
				for (long k = 0; k < steps - 1; ++k) {
					res[i + (idx << logN0)] = res[i + (((idx + r1) % N1) << logN0)];
					idx = (idx + r1) % N1;
				}
				res[i + (idx << logN0)] = tmp;
			}
		}
	}
}

void Ring::conjugate(ZZ* res, ZZ* p) {
	for (long j = 0; j < N; j += N0) {
		res[j] = p[j];
		for (long i = 1; i < N0; ++i) {
			res[N0 - i + j] = -p[i + j];
		}
	}
	for (long i = 0; i < N0; ++i) {
		for (long j = 0; j < Nh; j+=N0) {
			swap(res[i+j], res[i+j+Nh]);
		}
	}
}

//----------------------------------------------------------------------------------
//   SAMPLING
//----------------------------------------------------------------------------------


void Ring::sampleGauss(ZZ* res) {
	static double Pi = 4.0 * atan(1.0);
	static long const bignum = 0xfffffff;

	for (long i = 0; i < N; i+=2) {
		double r1 = (1 + RandomBnd(bignum)) / ((double)bignum + 1);
		double r2 = (1 + RandomBnd(bignum)) / ((double)bignum + 1);
		double theta=2 * Pi * r1;
		double rr= sqrt(-2.0 * log(r2)) * sigma;

		res[i] = (long) floor(rr * cos(theta) + 0.5);
		res[i + 1] = (long) floor(rr * sin(theta) + 0.5);
	}
}

void Ring::sampleHWT(ZZ* res) {
	long idx = 0;
	while(idx < h) {
		long i = RandomBnd(N);
		if(res[i] == 0) {
			res[i] = (rand()&1) ? ZZ(1) : ZZ(-1);
			idx++;
		}
	}
}

void Ring::sampleZO(ZZ* res) {
	for (long i = 0; i < N; ++i) {
		res[i] = (rand()&1) ? ZZ(0) : (rand()&1) ? ZZ(1) : ZZ(-1);
	}
}

void Ring::sampleUniform(ZZ* res, long bits) {
	for (long i = 0; i < N; i++) {
		res[i] = RandomBits_ZZ(bits);
	}
}
