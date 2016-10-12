#include "Utility\CUDA\CUDAManaged.cuh"
#include <cuda_runtime.h>

void* Managed::operator new(size_t len){
	void *ptr;
	cudaMallocManaged((void**)&ptr, len);
	cudaDeviceSynchronize();
	return ptr;
}

void Managed::operator delete(void *ptr) {
	cudaDeviceSynchronize();
	cudaFree(ptr);
}

