#include "Utility\CUDA\CUDAManaged.cuh"


void* Managed::operator new(size_t len){
	void *ptr;
	cudaMallocManaged(&ptr, len);
	cudaDeviceSynchronize();
	return ptr;
}

void Managed::operator delete(void *ptr) {
	cudaDeviceSynchronize();
	cudaFree(ptr);
}

