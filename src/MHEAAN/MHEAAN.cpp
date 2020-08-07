#include "TestScheme.h"

int main() {

//----------------------------------------------------------------------------------
//   STANDARD TESTS
//----------------------------------------------------------------------------------

//	TestScheme::testEncrypt(300, 30, 2, 2);
//	TestScheme::testStandard(300, 30, 2, 2);
//	TestScheme::testimult(300, 30, 2, 2);

//	TestScheme::testRotateFast(300, 30, 3, 3, 1, 0);
//	TestScheme::testConjugate(300, 30, 2, 2);

	TestScheme::testBootstrap(40, 33, 7, 8, 4, 4);
//	TestScheme::testCiphertextWriteAndRead(10, 65, 30, 2);
//	TestScheme::test();

	return 0;
}
