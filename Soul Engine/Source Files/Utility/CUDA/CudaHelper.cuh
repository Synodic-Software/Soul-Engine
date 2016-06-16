#pragma once

#include "Utility\CUDAIncludes.h"
#include <iostream>
#include "cuda.h"

#define CUDA_FUNCTION __host__ __device__

#define GPU_CORES 640
#define BLOCK_HEIGHT 4 //cores/smm /32
#define WARP_SIZE 32

#define CudaCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		std::cout << cudaGetErrorString(code) << file << line << std::endl;
		//if (abort) exit(code);
	}
}

CUDA_FUNCTION uint randHash(uint a);
inline __device__ int getGlobalIdx_1D_1D()
{
	return blockIdx.x *blockDim.x + threadIdx.x;
}

__device__ int warp_bcast(int v, int leader);
__device__ int lane_id(void);

// warp-aggregated atomic increment
__device__
int FastAtomicAdd(int *ctr);

