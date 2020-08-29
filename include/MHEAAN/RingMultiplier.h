#ifndef MHEAAN_RINGMULTIPLIER_H_
#define MHEAAN_RINGMULTIPLIER_H_

#include <NTL/BasicThreadPool.h>
#include <NTL/ZZ.h>
#include <vector>
#include "Params.h"

using namespace std;
using namespace NTL;

class RingMultiplier {
public:

	uint64_t* gM1Pows = new uint64_t[M1];
	uint64_t** rootM1DFTPows = new uint64_t*[nprimes];
	uint64_t** rootM1DFTPowsInv = new uint64_t*[nprimes];

	uint64_t* pVec = new uint64_t[nprimes];
	uint64_t* prVec = new uint64_t[nprimes];
	long* pTwok = new long[nprimes];

	uint64_t* pInvVec = new uint64_t[nprimes];

	uint64_t** scaledRootM0Pows = new uint64_t*[nprimes];
	uint64_t** scaledRootN1Pows = new uint64_t*[nprimes];

	uint64_t** scaledRootM0PowsInv = new uint64_t*[nprimes];
	uint64_t** scaledRootN1PowsInv = new uint64_t*[nprimes];

	uint64_t* scaledN0Inv = new uint64_t[nprimes];
	uint64_t* scaledN1Inv = new uint64_t[nprimes];

	_ntl_general_rem_one_struct** red_ss_array = new _ntl_general_rem_one_struct*[nprimes];
	mulmod_precon_t** coeffpinv_array = new mulmod_precon_t*[nprimes];
	ZZ* pProd = new ZZ[nprimes];
	ZZ* pProdh = new ZZ[nprimes];
	ZZ** pHat = new ZZ*[nprimes];
	uint64_t** pHatInvModp = new uint64_t*[nprimes];

	RingMultiplier();

	bool primeTest(uint64_t p);

	void arrayBitReverse(uint64_t* a, long n);

	void NTTX0(uint64_t* a, long index);
	void INTTX0(uint64_t* a, long index);
	void NTTPO2X1(uint64_t* a, long index);
	void INTTPO2X1(uint64_t* a, long index);
	void NTTX1(uint64_t* a, long index);
	void INTTX1(uint64_t* a, long index);
	void NTT(uint64_t* a, long index);
	void INTT(uint64_t* a, long index);

	void toNTTX0(uint64_t* ra, ZZ* a, long np);
	void toNTTX1(uint64_t* ra, ZZ* a, long np);
	void toNTT(uint64_t* ra, ZZ* a, long np);

	void addNTTAndEqual(uint64_t* ra, uint64_t* rb, long np);

	void reconstruct(ZZ* x, uint64_t* rx, long np, const ZZ& q);

	void multX0(ZZ* x, ZZ* a, ZZ* b, long np, const ZZ& q);
	void multX0AndEqual(ZZ* a, ZZ* b, long np, const ZZ& q);
	void multNTTX0(ZZ* x, ZZ* a, uint64_t* rb, long np, const ZZ& q);
	void multNTTX0AndEqual(ZZ* a, uint64_t* rb, long np, const ZZ& q);
	void multDNTTX0(ZZ* x, uint64_t* ra, uint64_t* rb, long np, const ZZ& q);

	void multX1(ZZ* x, ZZ* a, ZZ* b, long np, const ZZ& q);
	void multX1AndEqual(ZZ* a, ZZ* b, long np, const ZZ& q);
	void multNTTX1(ZZ* x, ZZ* a, uint64_t* rb, long np, const ZZ& q);
	void multNTTX1AndEqual(ZZ* a, uint64_t* rb, long np, const ZZ& q);
	void multDNTTX1(ZZ* x, uint64_t* ra, uint64_t* rb, long np, const ZZ& q);

	void mult(ZZ* x, ZZ* a, ZZ* b, long np, const ZZ& q);
	void multAndEqual(ZZ* a, ZZ* b, long np, const ZZ& q);
	void multNTT(ZZ* x, ZZ* a, uint64_t* rb, long np, const ZZ& q);
	void multNTTAndEqual(ZZ* a, uint64_t* rb, long np, const ZZ& q);
	void multDNTT(ZZ* x, uint64_t* ra, uint64_t* rb, long np, const ZZ& q);

	void square(ZZ* x, ZZ* a, long np, const ZZ& q);
	void squareAndEqual(ZZ* a, long np, const ZZ& q);
	void squareNTT(ZZ* x, uint64_t* ra, long np, const ZZ& q);

	void butt1(uint64_t& a1, uint64_t& a2, uint64_t p, uint64_t pInv, uint64_t W);
	void butt2(uint64_t& a1, uint64_t& a2, uint64_t p, uint64_t pInv, uint64_t W);
	void divByN(uint64_t& a, uint64_t p, uint64_t pInv, uint64_t NScaleInv);
	void mulMod(uint64_t& r, uint64_t a, uint64_t b, uint64_t p);
	void mulModBarrett(uint64_t& r, uint64_t a, uint64_t b, uint64_t p, uint64_t pr, long twok);
	void mulModBarrettAndEqual(uint64_t& r, uint64_t b, uint64_t p, uint64_t pr, long twok);

	uint64_t powerModulus(uint64_t x, uint64_t y, uint64_t p);

	uint64_t invThis(uint64_t x);

	uint32_t bitReverse(uint32_t x);

	void findPrimeFactors(vector<uint64_t> &s, uint64_t number);

	uint64_t findPrimitiveRoot(uint64_t m);

	uint64_t findMthRootOfUnity(uint64_t M, uint64_t p);

};

#endif /* RINGMULTIPLIER_H_ */
