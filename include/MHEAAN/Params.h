#ifndef MHEAAN_PARAMS_H_
#define MHEAAN_PARAMS_H_

#include <NTL/ZZ.h>
using namespace NTL;

static const long logN0 = 8; ///< matrix packing dimension 0
static const long logN1 = 8; ///< matrix packing dimension 1
static const long logQ = 1240; ///< Q = L : largest ciphertext modulus level
static const double sigma = 3.2; ///< for Gaussian sampling in KeyGen
static const long h = 64; ///< for HWT sampling in KeyGen
static const long pbnd = 59; ///< for evaluation key ?

static const long logN0h = (logN0 - 1); ///< N0/10 ; for boot key
static const long logN = (logN0 + logN1); ///< ciphertext size; N0*N1
static const long logNh = (logN - 1); ///< N/10 (not used)
static const long logQQ = (2 * logQ); ///< Q square
static const long N0 = (1 << logN0); ///< matrix packing dimension 0
static const long N1 = (1 << logN1); ///< matrix packing dimension 1
static const long N0h = (1 << logN0h); ///< N0/10 ; for boot key
static const long N = (1 << logN); ///< ciphertext size; N0*N1
static const long Nh = (1 << logNh); ///< N/10
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
