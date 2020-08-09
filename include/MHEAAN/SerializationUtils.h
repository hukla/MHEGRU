#ifndef MHEAAN_SERIALIZATIONUTILS_H_
#define MHEAAN_SERIALIZATIONUTILS_H_

#include <iostream>
#include <iosfwd>
#include <cstdint>
#include <fstream>
#include "Key.h"
#include "Ciphertext.h"

using namespace std;
using namespace NTL;

class SerializationUtils {
public:

	static void writeCiphertext(Ciphertext* ciphertext, string path);
	static Ciphertext& readCiphertext(string path);

	static void writeKey(Key* key, string path);
	static Key& readKey(string path);
};

#endif
