// #pragma once

// #include <cuda_runtime.h>
// #include <iostream>
// #include "Core/Utility/Types.h"

// #define WARP_SIZE 32

// #define CudaCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
// inline void gpuAssert(cudaError_t code, const char *file, int line)
// {
// 	if (code != cudaSuccess)
// 	{
// 		std::cout << cudaGetErrorString(code) << " "<< file << line << std::endl;
// 		throw "Cuda Error";//std::exception("Cuda Error");
// 	}

// 	if(cudaSuccess != cudaGetLastError()) {
// 		throw "CudaCheck on Something";//std::exception("Missed 'CudaCheck on Something");
// 	}
// }

// __host__ __device__ uint randHash(uint a);

// __device__ int warp_bcast(int v, int leader);
// __device__ int lane_id();

// // warp-aggregated atomic increment
// __device__
// int FastAtomicAdd(int *ctr);

// inline int _GetCoresPerMP(int major, int minor)
// {
// 	// Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
// 	typedef struct
// 	{
// 		int SM;  //0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
// 		int Cores;
// 	} sSMtoCores;

// 	sSMtoCores nGpuArchCoresPerSM[] =
// 	{
// 		{ 0x20, 32 },  //Fermi Generation (SM 2.0) GF100 class
// 		{ 0x21, 48 },  //Fermi Generation (SM 2.1) GF10x class
// 		{ 0x30, 192 }, //Kepler Generation (SM 3.0) GK10x class
// 		{ 0x32, 192 }, //Kepler Generation (SM 3.2) GK10x class
// 		{ 0x35, 192 }, //Kepler Generation (SM 3.5) GK11x class
// 		{ 0x37, 192 }, //Kepler Generation (SM 3.7) GK21x class
// 		{ 0x50, 128 }, //Maxwell Generation (SM 5.0) GM10x class
// 		{ 0x52, 128 }, //Maxwell Generation (SM 5.2) GM20x class
// 		{ 0x53, 128 }, //Maxwell Generation (SM 5.3) GM20x class
// 		{ 0x60, 64 },  //Pascal Generation (SM 6.0) GP100 class
// 		{ 0x61, 128 }, //Pascal Generation (SM 6.1) GP10x class
// 		{ 0x62, 128 }, //Pascal Generation (SM 6.2) GP10x class
// 		{ -1, -1 }
// 	};

// 	int index = 0;

// 	while (nGpuArchCoresPerSM[index].SM != -1)
// 	{
// 		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
// 		{
// 			return nGpuArchCoresPerSM[index].Cores;
// 		}

// 		index++;
// 	}

// 	//defaulted case, go back a version
// 	return nGpuArchCoresPerSM[index - 1].Cores;
// }


// inline int _GetBlocksPerMP(int major, int minor)
// {
// 	typedef struct
// 	{
// 		int CC;
// 		int Cores;
// 	} sSMtoCores;

// 	sSMtoCores nGpuArchCoresPerSM[] =
// 	{
// 		{ 0x20, 8 },  //Fermi Generation (CC 2.0) GF100 class
// 		{ 0x21, 8 },  //Fermi Generation (CC 2.1) GF10x class
// 		{ 0x30, 16 },  //Kepler Generation (CC 3.0) GK10x class
// 		{ 0x32, 16 },  //Kepler Generation (CC 3.2) GK10x class
// 		{ 0x35, 16 },  //Kepler Generation (CC 3.5) GK11x class
// 		{ 0x37, 16 },  //Kepler Generation (CC 3.7) GK21x class
// 		{ 0x50, 32 },  //Maxwell Generation (CC 5.0) GM10x class
// 		{ 0x52, 32 },  //Maxwell Generation (CC 5.2) GM20x class
// 		{ 0x53, 32 },  //Maxwell Generation (CC 5.3) GM20x class
// 		{ 0x60, 32 },  //Pascal Generation (CC 6.0) GP100 class
// 		{ 0x61, 32 },  //Pascal Generation (CC 6.1) GP10x class
// 		{ 0x62, 32 },  //Pascal Generation (CC 6.2) GP10x class
// 		{ -1, -1 }
// 	};

// 	int index = 0;

// 	while (nGpuArchCoresPerSM[index].CC != -1)
// 	{
// 		if (nGpuArchCoresPerSM[index].CC == ((major << 4) + minor))
// 		{
// 			return nGpuArchCoresPerSM[index].Cores;
// 		}

// 		index++;
// 	}

// 	//S_LOG_ERROR("Defaulted to a previous CC version");
// 	//defaulted case, go back a version
// 	return nGpuArchCoresPerSM[index - 1].Cores;
// }

// inline int _GetWarpsPerMP(int major, int minor)
// {
// 	typedef struct
// 	{
// 		int CC;
// 		int Cores;
// 	} sSMtoCores;

// 	sSMtoCores nGpuArchCoresPerSM[] =
// 	{
// 		{ 0x20, 48 },  //Fermi Generation (CC 2.0) GF100 class
// 		{ 0x21, 48 },  //Fermi Generation (CC 2.1) GF10x class
// 		{ 0x30, 64 },  //Kepler Generation (CC 3.0) GK10x class
// 		{ 0x32, 64 },  //Kepler Generation (CC 3.2) GK10x class
// 		{ 0x35, 64 },  //Kepler Generation (CC 3.5) GK11x class
// 		{ 0x37, 64 },  //Kepler Generation (CC 3.7) GK21x class
// 		{ 0x50, 64 },  //Maxwell Generation (CC 5.0) GM10x class
// 		{ 0x52, 64 },  //Maxwell Generation (CC 5.2) GM20x class
// 		{ 0x53, 64 },  //Maxwell Generation (CC 5.3) GM20x class
// 		{ 0x60, 64 },  //Pascal Generation (CC 6.0) GP100 class
// 		{ 0x61, 64 },  //Pascal Generation (CC 6.1) GP10x class
// 		{ 0x62, 64 },  //Pascal Generation (CC 6.2) GP10x class
// 		{ -1, -1 }
// 	};

// 	int index = 0;

// 	while (nGpuArchCoresPerSM[index].CC != -1)
// 	{
// 		if (nGpuArchCoresPerSM[index].CC == ((major << 4) + minor))
// 		{
// 			return nGpuArchCoresPerSM[index].Cores;
// 		}

// 		index++;
// 	}

// 	//S_LOG_ERROR("Defaulted to a previous CC version");
// 	//defaulted case, go back a version
// 	return nGpuArchCoresPerSM[index - 1].Cores;
// }

