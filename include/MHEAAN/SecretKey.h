#ifndef MHEAAN_SECRETKEY_H_
#define MHEAAN_SECRETKEY_H_

#include <NTL/ZZ.h>
#include "Params.h"
#include "Ring.h"

using namespace std;
using namespace NTL;

class SecretKey {
public:

	ZZ* sx = new ZZ[N]; ///< secret key

	SecretKey(Ring& ring);
	SecretKey();

};

#endif
