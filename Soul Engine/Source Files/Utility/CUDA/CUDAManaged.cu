#include "Utility\CUDA\CUDAManaged.cuh"
#include <cuda_runtime.h>
#include "Utility\CUDA\CUDAHelper.cuh" 

void* Managed::operator new(size_t len){
	void *ptr;
	CudaCheck(cudaMallocManaged((void**)&ptr, len));
	CudaCheck(cudaDeviceSynchronize());
	return ptr;
}

void Managed::operator delete(void *ptr) {
	CudaCheck(cudaDeviceSynchronize());
	CudaCheck(cudaFree(ptr));
}

