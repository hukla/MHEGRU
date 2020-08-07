#include "TestScheme.h"

#include <NTL/BasicThreadPool.h>
#include <NTL/RR.h>
#include <NTL/ZZ.h>

#include "Ciphertext.h"
#include "EvaluatorUtils.h"
#include "Ring.h"
#include "Scheme.h"
#include "SecretKey.h"
#include "StringUtils.h"
#include "TimeUtils.h"

using namespace std;
using namespace NTL;


//----------------------------------------------------------------------------------
//   STANDARD TESTS
//----------------------------------------------------------------------------------


void TestScheme::testEncrypt(long logq, long logp, long logn0, long logn1) {
	cout << "!!! START TEST ENCRYPT !!!" << endl;

	srand(time(NULL));
	SetNumThreads(8);

	TimeUtils timeutils;
	Ring ring;
	SecretKey secretKey(ring);
	Scheme scheme(secretKey, ring);

	long n0 = (1 << logn0);
	long n1 = (1 << logn1);
	long n = n0 * n1;

	complex<double>* mmat = EvaluatorUtils::randomComplexSignedArray(n);

	timeutils.start("Encode matrix");
	Plaintext msg;
	scheme.encode(msg, mmat, n0, n1, logp);
	timeutils.stop("Encode matrix");

	timeutils.start("Encrypt msg");
	Ciphertext cipher;
	scheme.encryptMsg(cipher, msg, logQ);
	timeutils.stop("Encrypt msg");

	timeutils.start("Decrypt msg");
	Plaintext dsg;
	scheme.decryptMsg(dsg, secretKey, cipher);
	timeutils.stop("Decrypt msg");

	timeutils.start("Decode matrix");
	complex<double>* dmat = scheme.decode(dsg);
	timeutils.stop("Decode matrix");

	StringUtils::compare(mmat, dmat, n, "val");

	cout << "!!! END TEST ENCRYPT !!!" << endl;
}

void TestScheme::testStandard(long logq, long logp, long logn0, long logn1) {
	cout << "!!! START TEST STANDARD !!!" << endl;

	srand(time(NULL));
	SetNumThreads(8);

	TimeUtils timeutils;
	Ring ring;
	SecretKey secretKey(ring);
	Scheme scheme(secretKey, ring);

	long n0 = (1 << logn0);
	long n1 = (1 << logn1);
	long n = n0 * n1;

	complex<double>* mmat1 = EvaluatorUtils::randomComplexSignedArray(n);
	complex<double>* mmat2 = EvaluatorUtils::randomComplexSignedArray(n);
	complex<double>* madd = new complex<double>[n];
	complex<double>* mmult = new complex<double>[n];
	for (long i = 0; i < n; ++i) {
		mmult[i] = mmat1[i] * mmat2[i];
		madd[i] = mmat1[i] + mmat2[i];
	}
	Ciphertext cipher1, cipher2;
	scheme.encrypt(cipher1, mmat1, n0, n1, logp, logq);
	scheme.encrypt(cipher2, mmat2, n0, n1, logp, logq);

	Ciphertext cadd(cipher1);
	timeutils.start("add matrix");
	scheme.addAndEqual(cadd, cipher2);
	timeutils.stop("add matrix");

	Ciphertext cmult(cipher1);
	timeutils.start("mult matrix");
	scheme.multAndEqual(cmult, cipher2);
	scheme.reScaleByAndEqual(cmult, logp);
	timeutils.stop("mult matrix");

	complex<double>* dadd = scheme.decrypt(secretKey, cadd);
	complex<double>* dmult = scheme.decrypt(secretKey, cmult);

	StringUtils::compare(madd, dadd, n, "add");
	StringUtils::compare(mmult, dmult, n, "mult");

	cout << "!!! END TEST STANDARD !!!" << endl;
}

void TestScheme::testimult(long logq, long logp, long logn0, long logn1) {
	cout << "!!! START TEST i MULTIPLICATION !!!" << endl;

	srand(time(NULL));
	SetNumThreads(8);

	TimeUtils timeutils;
	Ring ring;
	SecretKey secretKey(ring);
	Scheme scheme(secretKey, ring);

	long n0 = (1 << logn0);
	long n1 = (1 << logn1);
	long n = n0 * n1;

	complex<double>* mmat = EvaluatorUtils::randomComplexSignedArray(n);
	complex<double>* mmatimult = new complex<double>[n];
	for (long i = 0; i < n; ++i) {
		mmatimult[i].real(-mmat[i].imag());
		mmatimult[i].imag(mmat[i].real());
	}

	Ciphertext cipher;
	scheme.encrypt(cipher, mmat, n0, n1, logp, logq);

	timeutils.start("Multiplication by i");
	scheme.imultAndEqual(cipher);
	timeutils.stop("Multiplication by i");

	complex<double>* dmatimult = scheme.decrypt(secretKey, cipher);

	StringUtils::compare(mmatimult, dmatimult, n, "imult");

	cout << "!!! END TEST i MULTIPLICATION !!!" << endl;
}


//----------------------------------------------------------------------------------
//   ROTATION & CONJUGATION & TRANSPOSITION TESTS
//----------------------------------------------------------------------------------


void TestScheme::testRotateFast(long logq, long logp, long logn0, long logn1, long r0, long r1) {
	cout << "!!! START TEST ROTATE FAST !!!" << endl;

	srand(time(NULL));
	SetNumThreads(8);

	TimeUtils timeutils;
	Ring ring;
	SecretKey secretKey(ring);
	Scheme scheme(secretKey, ring);

	scheme.addLeftRotKey(secretKey, r0, r1);

	long n0 = (1 << logn0);
	long n1 = (1 << logn1);
	long n = n0 * n1;

	complex<double>* mmat = EvaluatorUtils::randomComplexSignedArray(n);
	Ciphertext cipher;
	scheme.encrypt(cipher, mmat, n0, n1, logp, logq);

	timeutils.start("Left rotate fast");
	scheme.leftRotateAndEqual(cipher, r0, r1);
	timeutils.stop("Left rotate fast");

	complex<double>* dmat = scheme.decrypt(secretKey, cipher);
	EvaluatorUtils::leftRotateAndEqual(mmat, n0, n1, r0, r1);
	StringUtils::compare(mmat, dmat, n, "val");

	cout << "!!! END TEST ROTATE FAST !!!" << endl;
}

void TestScheme::testConjugate(long logq, long logp, long logn0, long logn1) {
	cout << "!!! START TEST CONJUGATE !!!" << endl;

	srand(time(NULL));
	SetNumThreads(8);

	TimeUtils timeutils;
	Ring ring;
	SecretKey secretKey(ring);
	Scheme scheme(secretKey, ring);

	scheme.addConjKey(secretKey);

	long n0 = (1 << logn0);
	long n1 = (1 << logn1);
	long n = n0 * n1;

	complex<double>* mmat = EvaluatorUtils::randomComplexSignedArray(n);
	complex<double>* mmatconj = new complex<double>[n];
	for (long i = 0; i < n; ++i) {
		mmatconj[i] = conj(mmat[i]);
	}

	Ciphertext cipher;
	scheme.encrypt(cipher, mmat, n0, n1, logp, logq);

	timeutils.start("Conjugate");
	scheme.conjugateAndEqual(cipher);
	timeutils.stop("Conjugate");

	complex<double>* dmatconj = scheme.decrypt(secretKey, cipher);
	StringUtils::compare(mmatconj, dmatconj, n, "conj");

	cout << "!!! END TEST CONJUGATE !!!" << endl;
}

void TestScheme::testBootstrap(long logq, long logp, long logn0, long logn1, long logT, long logI) {
	cout << "!!! START TEST BOOTSTRAP !!!" << endl;

	srand(time(NULL));
	SetNumThreads(8);

	TimeUtils timeutils;
	timeutils.start("Scheme generating");
	Ring ring;
	SecretKey secretKey(ring);
	Scheme scheme(secretKey, ring);
	timeutils.stop("Scheme generated");

	timeutils.start("BootKey generating");
	scheme.addBootKey(secretKey, logn0, logn1, logq + logI);
	timeutils.stop("BootKey generated");

	long n0 = (1 << logn0);
	long n1 = (1 << logn1);
	long n = n0 * n1;

	complex<double>* mmat = EvaluatorUtils::randomComplexSignedArray(n);

	Ciphertext cipher;
	scheme.encrypt(cipher, mmat, n0, n1, logp, logq);

	cout << "cipher logq before: " << cipher.logq << endl;
	scheme.normalizeAndEqual(cipher);

	cipher.logq = logQ;
	cipher.logp = logq + logI;

	timeutils.start("Coeff to Slot");
	scheme.coeffToSlotAndEqual(cipher);
	timeutils.stop("Coeff to Slot");

	timeutils.start("Remove I Part");
	scheme.removeIPartAndEqual(cipher, logT, logI);
	timeutils.stop("Remove I Part");

	timeutils.start("Slot to Coeff");
	scheme.slotToCoeffAndEqual(cipher);
	timeutils.stop("Slot to Coeff");

	cout << "cipher logp after: " << cipher.logp << endl;
	cout << "cipher logq after: " << cipher.logq << endl;

	cipher.logp = logp;

	complex<double>* dmat = scheme.decrypt(secretKey, cipher);
	StringUtils::compare(mmat, dmat, 10, "boot");

	cout << "!!! END TEST BOOTSRTAP !!!" << endl;
}

