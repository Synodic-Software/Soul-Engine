#include "CUDADevice.h"
#include <cuda_runtime_api.h>
#include "Utility/CUDA/CudaHelper.cuh"

/*
 *    Constructor.
 *    @param	o	An uint to process.
 */

CUDADevice::CUDADevice(uint deviceNumber) :
GPUDevice(deviceNumber)
{

	api = CUDA;

	CudaCheck(cudaSetDevice(deviceNumber));
	CudaCheck(cudaGetDeviceProperties(&deviceProperties, deviceNumber));
	CudaCheck(cudaSetDevice(0));

}

/* Destructor. */
CUDADevice::~CUDADevice() {

	cudaSetDevice(order);
	cudaDeviceReset();

}

/*
*    Gets core count.
*    @return	The core count.
*/

int CUDADevice::GetCoreCount() {
	int device;
	CudaCheck(cudaGetDevice(&device));

	return _GetCoresPerMP(deviceProperties.major, deviceProperties.minor) * deviceProperties.multiProcessorCount;
}

/*
*    Gets sm count.
*    @return	The sm count.
*/

int CUDADevice::GetSMCount() {
	int device;
	CudaCheck(cudaGetDevice(&device));

	return deviceProperties.multiProcessorCount;
}

/*
*    Gets warp size.
*    @return	The warp size.
*/

int CUDADevice::GetWarpSize() {
	int device;
	CudaCheck(cudaGetDevice(&device));

	return deviceProperties.warpSize;
}

/*
*    Gets warps per mp.
*    @return	The warps per mp.
*/

int CUDADevice::GetWarpsPerMP() {
	int device;
	CudaCheck(cudaGetDevice(&device));

	return _GetWarpsPerMP(deviceProperties.major, deviceProperties.minor);
}

/*
*    Gets blocks per mp.
*    @return	The blocks per mp.
*/

int CUDADevice::GetBlocksPerMP() {
	int device;
	CudaCheck(cudaGetDevice(&device));

	return _GetBlocksPerMP(deviceProperties.major, deviceProperties.minor);
}