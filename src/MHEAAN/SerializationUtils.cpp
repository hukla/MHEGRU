#include "SerializationUtils.h"

void SerializationUtils::writeCiphertext(Ciphertext* cipher, string path) {
	fstream fout;
	fout.open(path, ios::binary|ios::out);
	long n0 = cipher->n0;
	long n1 = cipher->n1;
	long logp = cipher->logp;
	long logq = cipher->logq;
	fout.write(reinterpret_cast<char*>(&n0), sizeof(long));
	fout.write(reinterpret_cast<char*>(&n1), sizeof(long));
	fout.write(reinterpret_cast<char*>(&logp), sizeof(long));
	fout.write(reinterpret_cast<char*>(&logq), sizeof(long));

	long np = ceil(((double)logq + 1)/8);
	unsigned char* bytes = new unsigned char[np];
	ZZ q = conv<ZZ>(1) << logq;
	for (long i = 0; i < N; ++i) {
		cipher->ax[i] %= q;
		BytesFromZZ(bytes, cipher->ax[i], np);
		fout.write(reinterpret_cast<char*>(bytes), np);
	}
	for (long i = 0; i < N; ++i) {
		cipher->bx[i] %= q;
		BytesFromZZ(bytes, cipher->bx[i], np);
		fout.write(reinterpret_cast<char*>(bytes), np);
	}
	fout.close();
}

Ciphertext& SerializationUtils::readCiphertext(string path) {
	long n0, n1, logp, logq;
	fstream fin;
	fin.open(path, ios::binary|ios::in);
	fin.read(reinterpret_cast<char*>(&n0), sizeof(long));
	fin.read(reinterpret_cast<char*>(&n1), sizeof(long));
	fin.read(reinterpret_cast<char*>(&logp), sizeof(long));
	fin.read(reinterpret_cast<char*>(&logq), sizeof(long));

	long np = ceil(((double)logq + 1)/8);
	unsigned char* bytes = new unsigned char[np];

	Ciphertext* res = new Ciphertext(logp, logq, n0, n1);
	for (long i = 0; i < N; ++i) {
		fin.read(reinterpret_cast<char*>(bytes), np);
		ZZFromBytes(res->ax[i], bytes, np);
	}
	for (long i = 0; i < N; ++i) {
		fin.read(reinterpret_cast<char*>(bytes), np);
		ZZFromBytes(res->bx[i], bytes, np);
	}
	fin.close();
	return *res;
}

void SerializationUtils::writeKey(Key* key, string path) {
	fstream fout;
	fout.open(path, ios::binary|ios::out);
	fout.write(reinterpret_cast<char*>(key->rax), Nnprimes * sizeof(uint64_t));
	fout.write(reinterpret_cast<char*>(key->rbx), Nnprimes * sizeof(uint64_t));
	fout.close();
}

Key& SerializationUtils::readKey(string path) {
	long np;
	Key* key = new Key();
	fstream fin;
	fin.open(path, ios::binary|ios::in);
	fin.read(reinterpret_cast<char*>(key->rax), (Nnprimes)*sizeof(uint64_t));
	fin.read(reinterpret_cast<char*>(key->rbx), (Nnprimes)*sizeof(uint64_t));
	fin.close();
	return *key;
}

