#include "Utility\CUDA\CUDAManaged.cuh"


void* Managed::operator new(size_t len){
	void *ptr;
	CudaCheck(cudaMallocManaged((void**)&ptr, len));
	cudaDeviceSynchronize();
	return ptr;
}

void Managed::operator delete(void *ptr) {
	cudaDeviceSynchronize();
	CudaCheck(cudaFree(ptr));
}

