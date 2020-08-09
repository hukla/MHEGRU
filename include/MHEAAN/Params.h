#ifndef MHEAAN_PARAMS_H_
#define MHEAAN_PARAMS_H_

#include <NTL/ZZ.h>
using namespace NTL;

static const long logN0 = 8;
static const long logN1 = 8;
static const long logQ = 1240; // Q = L : largest ciphertext modulus level
static const double sigma = 3.2;
static const long h = 64;
static const long pbnd = 59;

static const long logN0h = (logN0 - 1);
static const long logN = (logN0 + logN1);
static const long logNh = (logN - 1);
static const long logQQ = (2 * logQ);
static const long N0 = (1 << logN0);
static const long N1 = (1 << logN1);
static const long N0h = (1 << logN0h);
static const long N = (1 << logN);
static const long Nh = (1 << logNh);
static const long M0 = (N0 << 1);
static const long M1 = (N1 + 1);
static const long nprimes = (logQQ * 2 + logN + 3 + pbnd - 1) / pbnd;
static const long N0nprimes = (nprimes << logN0);
static const long N1nprimes = (nprimes << logN1);
static const long Nnprimes = (nprimes << logN);

static const long cbnd = (logQQ + NTL_ZZ_NBITS - 1) / NTL_ZZ_NBITS;

static const ZZ Q = power2_ZZ(logQ);
static const ZZ QQ = power2_ZZ(logQQ);

#endif
