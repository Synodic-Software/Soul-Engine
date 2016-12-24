#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "Metrics.h"
#include <iostream>


//#include <cuda.h>  
////#include <cuda_runtime_api.h>
//
//#include "Utility\Includes\GLMIncludes.h"
////#include <cuda_runtime.h>
////#include <device_launch_parameters.h>
////#include <device_functions.h>
//#include <curand.h>
//#include <curand_kernel.h>
//#include <vector_types.h>
//#include <driver_functions.h>
//#include "CUDA\CUDAMath.h"
//#include <cuda_profiler_api.h>
//#include "Metrics.h"


#define WARP_SIZE 32

#define CudaCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool = true)
{
	if (code != cudaSuccess)
	{
		std::cout << cudaGetErrorString(code) << file << line << std::endl;
		//if (abort) exit(code);
	}
}

__host__ __device__ uint randHash(uint a);
inline __device__ int getGlobalIdx_1D_1D()
{
	return blockIdx.x *blockDim.x + threadIdx.x;
}

__device__ int warp_bcast(int v, int leader);
__device__ int lane_id();

// warp-aggregated atomic increment
__device__
int FastAtomicAdd(int *ctr);


inline int _ConvertSMVer2Cores(int major, int minor)
{
	// Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
	typedef struct
	{
		int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] =
	{
		{ 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
		{ 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
		{ 0x30, 192 }, // Kepler Generation (SM 3.0) GK10x class
		{ 0x32, 192 }, // Kepler Generation (SM 3.2) GK10x class
		{ 0x35, 192 }, // Kepler Generation (SM 3.5) GK11x class
		{ 0x37, 192 }, // Kepler Generation (SM 3.7) GK21x class
		{ 0x50, 128 }, // Maxwell Generation (SM 5.0) GM10x class
		{ 0x52, 128 }, // Maxwell Generation (SM 5.2) GM20x class
		{ 0x53, 128 }, // Maxwell Generation (SM 5.3) GM20x class
		{ 0x60, 64 }, // Pascal Generation (SM 6.0) GP100 class
		{ 0x61, 128 }, // Pascal Generation (SM 6.1) GP10x class
		{ 0x62, 128 }, // Pascal Generation (SM 6.2) GP10x class
		{ -1, -1 }
	};

	int index = 0;

	while (nGpuArchCoresPerSM[index].SM != -1)
	{
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
		{
			return nGpuArchCoresPerSM[index].Cores;
		}

		index++;
	}

	// If we don't find the values, we default use the previous one to run properly
	//LOG(TRACE, "MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[index - 1].Cores);
	return nGpuArchCoresPerSM[index - 1].Cores;
}
// end of GPU Architecture definitions
