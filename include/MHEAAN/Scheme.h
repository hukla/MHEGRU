#ifndef MHEAAN_SCHEME_H_
#define MHEAAN_SCHEME_H_

#include <NTL/RR.h>
#include <NTL/ZZ.h>

#include "SecretKey.h"
#include "Ciphertext.h"
#include "Plaintext.h"
#include "Key.h"
#include "EvaluatorUtils.h"
#include "Ring.h"

using namespace std;
using namespace NTL;

static long ENCRYPTION = 0;
static long MULTIPLICATION  = 1;
static long CONJUGATION = 2;

class Scheme {
private:
public:

	bool isSerialized;

	Ring* ring;

	map<long, Key&> keyMap;
	map<pair<long, long>, Key&> leftRotKeyMap;

	map<long, string> serKeyMap;
	map<pair<long, long>, string> serLeftRotKeyMap;


	Scheme(SecretKey& secretKey, Ring& ring, bool isSerialized = false);

    Scheme() {};


    //----------------------------------------------------------------------------------
	//   KEYS GENERATION
	//----------------------------------------------------------------------------------


	void addEncKey(SecretKey& secretKey);

	void addMultKey(SecretKey& secretKey);

	void addConjKey(SecretKey& secretKey);

	void addLeftRotKey(SecretKey& secretKey, long r0, long r1);

	void addLeftX0RotKeys(SecretKey& secretKey);

	void addLeftX1RotKeys(SecretKey& secretKey);

	void addRightX0RotKeys(SecretKey& secretKey);

	void addRightX1RotKeys(SecretKey& secretKey);

	void addBootKey(SecretKey& secretKey, long logn0, long logn1, long logp);

	void addSqrMatKeys(SecretKey& secretKey, long logn, long logp);

	void addTransposeKeys(SecretKey& secretKey, long logn, long logp);

	//----------------------------------------------------------------------------------
	//   ENCODING & DECODING
	//----------------------------------------------------------------------------------


	void encode(Plaintext& res, complex<double>* vals, long n0, long n1, long logp);

	void encode(Plaintext& res, double* vals, long n0, long n1, long logp);

	complex<double>* decode(Plaintext& msg);


	//----------------------------------------------------------------------------------
	//   ENCRYPTION & DECRYPTION
	//----------------------------------------------------------------------------------


	void encryptMsg(Ciphertext& res, Plaintext& msg, long logq);

	void encrypt(Ciphertext& res, complex<double>* vals, long n0, long n1, long logp, long logq);

	void encrypt(Ciphertext& res, double* vals, long n0, long n1, long logp, long logq);

	void decryptMsg(Plaintext& res, SecretKey& secretKey, Ciphertext& cipher);

	complex<double>* decrypt(SecretKey& secretKey, Ciphertext& cipher);


	//----------------------------------------------------------------------------------
	//   HOMOMORPHIC OPERATIONS
	//----------------------------------------------------------------------------------


	void negate(Ciphertext& res, Ciphertext& cipher);
	void negateAndEqual(Ciphertext& cipher);

	void add(Ciphertext& res, Ciphertext& cipher1, Ciphertext& cipher2);
	void addAndEqual(Ciphertext& cipher1, Ciphertext& cipher2);

	void addConst(Ciphertext& res, Ciphertext& cipher, double cnst, long logp = -1);
	void addConst(Ciphertext& res, Ciphertext& cipher, RR& cnst, long logp = -1);

	void addConstAndEqual(Ciphertext& cipher, RR& cnst, long logp = -1);
	void addConstAndEqual(Ciphertext& cipher, double cnst, long logp = -1);

	void addPoly(Ciphertext& res, Ciphertext& cipher, ZZ* poly, long logp);
	void addPolyAndEqual(Ciphertext& cipher, ZZ* poly, long logp);

	void sub(Ciphertext& res, Ciphertext& cipher1, Ciphertext& cipher2);
	void subAndEqual(Ciphertext& cipher1, Ciphertext& cipher2);
	void subAndEqual2(Ciphertext& cipher1, Ciphertext& cipher2);

	void imult(Ciphertext& res, Ciphertext& cipher);
	void imultAndEqual(Ciphertext& cipher);

	void idiv(Ciphertext& res, Ciphertext& cipher);
	void idivAndEqual(Ciphertext& cipher);

	void mult(Ciphertext& res, Ciphertext& cipher1, Ciphertext& cipher2);
	void multAndEqual(Ciphertext& cipher1, Ciphertext& cipher2);

	void square(Ciphertext& res, Ciphertext& cipher);
	void squareAndEqual(Ciphertext& cipher);

	void multConst(Ciphertext& res, Ciphertext& cipher, RR& cnst, long logp);
	void multConst(Ciphertext& res, Ciphertext& cipher, double cnst, long logp);
	void multConst(Ciphertext& res, Ciphertext& cipher, complex<double> cnst, long logp);

	void multConstAndEqual(Ciphertext& cipher, RR& cnst, long logp);
	void multConstAndEqual(Ciphertext& cipher, double cnst, long logp);
	void multConstAndEqual(Ciphertext& cipher, complex<double> cnst, long logp);

	void multPolyX0(Ciphertext& res, Ciphertext& cipher, ZZ* poly, long logp);
	void multPolyX0AndEqual(Ciphertext& cipher, ZZ* poly, long logp);
	void multPolyNTTX0(Ciphertext& res, Ciphertext& cipher, uint64_t* rpoly, long bnd, long logp);
	void multPolyNTTX0AndEqual(Ciphertext& cipher, uint64_t* rpoly, long bnd, long logp);

	void multPolyX1(Ciphertext& res, Ciphertext& cipher, ZZ* rpoly, ZZ* ipoly, long logp);
	void multPolyX1AndEqual(Ciphertext& cipher, ZZ* rpoly, ZZ* ipoly, long logp);

	void multPoly(Ciphertext& res, Ciphertext& cipher, ZZ* poly, long logp);
	void multPolyAndEqual(Ciphertext& cipher, ZZ* poly, long logp);

	void multPolyNTT(Ciphertext& res, Ciphertext& cipher, uint64_t* rpoly, long bnd, long logp);
	void multPolyNTTAndEqual(Ciphertext& cipher, uint64_t* rpoly, long bnd, long logp);

	void multByMonomial(Ciphertext& res, Ciphertext& cipher, const long d0, const long d1);
	void multByMonomialAndEqual(Ciphertext& cipher, const long d0, const long d1);

	void multPo2(Ciphertext& res, Ciphertext& cipher, long bits);
	void multPo2AndEqual(Ciphertext& cipher, long bits);

	void divPo2(Ciphertext& res, Ciphertext& cipher, long logd);
	void divPo2AndEqual(Ciphertext& cipher, long logd);


	//----------------------------------------------------------------------------------
	//   RESCALING & MODULUS DOWN
	//----------------------------------------------------------------------------------


	void reScaleBy(Ciphertext& res, Ciphertext& cipher, long dlogq);
	void reScaleTo(Ciphertext& res, Ciphertext& cipher, long logq);

	void reScaleByAndEqual(Ciphertext& cipher, long dlogq);
	void reScaleToAndEqual(Ciphertext& cipher, long logq);

	void modDownBy(Ciphertext& res, Ciphertext& cipher, long dlogq);
	void modDownTo(Ciphertext& res, Ciphertext& cipher, long logq);

	void modDownByAndEqual(Ciphertext& cipher, long dlogq);
	void modDownToAndEqual(Ciphertext& cipher, long logq);


	//----------------------------------------------------------------------------------
	//   ROTATIONS & CONJUGATIONS
	//----------------------------------------------------------------------------------

	void leftRotate(Ciphertext& res, Ciphertext& cipher, long r0, long r1);
	void rightRotate(Ciphertext& res, Ciphertext& cipher, long r0, long r1);

	void leftRotateAndEqual(Ciphertext& cipher, long r0, long r1);
	void rightRotateAndEqual(Ciphertext& cipher, long r0, long r1);

	void conjugate(Ciphertext& res, Ciphertext& cipher);
	void conjugateAndEqual(Ciphertext& cipher);


	//----------------------------------------------------------------------------------
	//   BOOTSTRAPPING
	//----------------------------------------------------------------------------------


	void normalizeAndEqual(Ciphertext& cipher);

	void coeffToSlotX0AndEqual(Ciphertext& cipher);
	void coeffToSlotX1AndEqual(Ciphertext& cipher);
	void coeffToSlotAndEqual(Ciphertext& cipher);

	void slotToCoeffX0AndEqual(Ciphertext& cipher);
	void slotToCoeffX1AndEqual(Ciphertext& cipher);
	void slotToCoeffAndEqual(Ciphertext& cipher);

	void exp2piAndEqual(Ciphertext& cipher, long logp);

	void removeIPartAndEqual(Ciphertext& cipher, long logT, long logI = 4);

	void bootstrapX0AndEqual(Ciphertext& cipher, long logq, long logQ, long logT, long logI = 4);
	void bootstrapX1AndEqual(Ciphertext& cipher, long logq, long logQ, long logT, long logI = 4);
	void bootstrapAndEqual(Ciphertext& cipher, long logq, long logQ, long logT, long logI = 4);

};

#endif
