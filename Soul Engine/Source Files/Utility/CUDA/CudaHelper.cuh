#pragma once

#include "Utility\CUDAIncludes.h"
#include <iostream>

#define CUDA_FUNCTION __host__ __device__

#define CudaCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		std::cout<<cudaGetErrorString(code)<< file<< line<<std::endl;
		if (abort) exit(code);
	}
}

CUDA_FUNCTION uint randHash(uint a);
inline __device__ int getGlobalIdx_1D_1D()
{
	return blockIdx.x *blockDim.x + threadIdx.x;
}