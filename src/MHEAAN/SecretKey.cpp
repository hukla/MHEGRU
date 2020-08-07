#include "SecretKey.h"

SecretKey::SecretKey(Ring& ring) {
	ring.sampleHWT(sx);
}

SecretKey::SecretKey() {}
