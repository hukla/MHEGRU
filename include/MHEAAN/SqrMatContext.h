#ifndef MPHEAAN_MATRIXCONTEXT_H_
#define MPHEAAN_MATRIXCONTEXT_H_

#include <NTL/ZZ.h>

using namespace NTL;

class SqrMatContext {
public:

	ZZ** mvec;
	long* bndVec;
	long logp;

	SqrMatContext(ZZ** mvec, long logp);
};

#endif /* MATRIXCONTEXT_H_ */
