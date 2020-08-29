#ifndef MHEAAN_TESTSCHEME_H_
#define MHEAAN_TESTSCHEME_H_

class TestScheme {
public:


	//----------------------------------------------------------------------------------
	//   STANDARD TESTS
	//----------------------------------------------------------------------------------


	static void testEncrypt(long logq, long logp, long logn0, long logn1);

	static void testStandard(long logq, long logp, long logn0, long logn1);

	static void testimult(long logq, long logp, long logn0, long logn1);

	static void testRotateFast(long logq, long logp, long logn0, long logn1, long r0, long r1);

	static void testConjugate(long logq, long logp, long logn0, long logn1);

	static void testBootstrap(long logq, long logp, long logn0, long logn1, long logT, long logI);

};

#endif
