#include "BootContext.h"

BootContext::BootContext(uint64_t** rpxVec,  uint64_t** rpxInvVec, uint64_t* rp1, uint64_t* rp2,
		long* bndVec, long* bndInvVec, long bnd1, long bnd2, long logp)
			: rpxVec(rpxVec), rpxInvVec(rpxInvVec), rp1(rp1), rp2(rp2),
			  bndVec(bndVec), bndInvVec(bndInvVec), bnd1(bnd1), bnd2(bnd2), logp(logp) {
}
