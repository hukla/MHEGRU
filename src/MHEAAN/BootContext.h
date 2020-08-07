#ifndef MPHEAAN_BOOTCONTEXT_H_
#define MPHEAAN_BOOTCONTEXT_H_

#include <NTL/ZZ.h>

using namespace NTL;

class BootContext {
public:

	uint64_t** rpxVec;
	uint64_t** rpxInvVec;
	uint64_t* rp1;
	uint64_t* rp2;

	long* bndVec;
	long* bndInvVec;
	long bnd1;
	long bnd2;

	long logp;

	BootContext(uint64_t** rpxVec = NULL, uint64_t** rpxInvVec = NULL, uint64_t* rp1 = NULL, uint64_t* rp2 = NULL,
			long* bndVec = NULL, long* bndInvVec = NULL, long bnd1 = 0, long bnd2 = 0, long logp = 0);

};

#endif /* BOOTCONTEXT_H_ */
