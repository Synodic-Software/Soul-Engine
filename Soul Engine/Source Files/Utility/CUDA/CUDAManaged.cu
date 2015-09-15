#ifndef Managed_H 
#define Managed_H

#include "Engine Core\BasicDependencies.h"

class Managed
{
public:
	void *operator new(size_t len){
		void *ptr;
		cudaMallocManaged(&ptr, len);
		cudaDeviceSynchronize();
		return ptr;
	}

		void operator delete(void *ptr) {
		cudaDeviceSynchronize();
		cudaFree(ptr);
	}
};

#endif