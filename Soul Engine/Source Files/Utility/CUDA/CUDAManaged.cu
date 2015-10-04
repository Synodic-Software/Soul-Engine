#pragma once

#include "Engine Core\BasicDependencies.h"
#include "Utility\CUDA\HelperClasses.cuh"

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
