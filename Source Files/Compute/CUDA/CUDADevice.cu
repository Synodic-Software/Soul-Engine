#include "CUDADevice.cuh"

/*
 *    Constructor.
 *    @param	o	An uint to process.
 */

CUDADevice::CUDADevice(int id_) :
	AbstractComputeDevice()
{
	deviceID = id_;

	CudaCheck(cudaSetDevice(deviceID));
	CudaCheck(cudaGetDeviceProperties(&deviceProperties, deviceID));

}

/* Destructor. */
CUDADevice::~CUDADevice() {

	cudaSetDevice(deviceID);
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

GPUExecutePolicy CUDADevice::BestExecutePolicy(const void* kernel, int(* func)(int))
{

	int blockSizeOut;
	int minGridSize;

	CudaCheck(
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(
		&minGridSize, &blockSizeOut, kernel, func,
		GetSMCount()*GetBlocksPerMP())
	);

	//get the device stats for persistant threads
	const int warpPerBlock = blockSizeOut / GetWarpSize();

	return GPUExecutePolicy(glm::vec3(minGridSize, 1, 1), glm::vec3(GetWarpSize(), warpPerBlock, 1), warpPerBlock * sizeof(int), 0);
}