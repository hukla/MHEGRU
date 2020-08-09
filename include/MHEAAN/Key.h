#ifndef MPHEAAN_KEY_H_
#define MPHEAAN_KEY_H_

#include "Params.h"
#include <NTL/ZZ.h>

using namespace NTL;

class Key {
public:

	uint64_t* rax = new uint64_t[Nnprimes];
	uint64_t* rbx = new uint64_t[Nnprimes];

	Key();

	virtual ~Key();
};

#endif
