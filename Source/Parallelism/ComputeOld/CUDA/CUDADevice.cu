#include "CUDADevice.cuh"


//CUDADevice::CUDADevice(int id_) :
//	AbstractComputeDevice()
//{
//	deviceID = id_;
//
//	CudaCheck(cudaSetDevice(deviceID));
//	CudaCheck(cudaGetDeviceProperties(&deviceProperties, deviceID));
//
//}
//
//CUDADevice::~CUDADevice() {
//
//	/*cudaSetDevice(deviceID);
//	cudaDeviceReset();*/
//
//}
//
//int CUDADevice::GetCoreCount() {
//	int device;
//	CudaCheck(cudaGetDevice(&device));
//
//	return _GetCoresPerMP(deviceProperties.major, deviceProperties.minor) * deviceProperties.multiProcessorCount;
//}
//
//int CUDADevice::GetSMCount() {
//	int device;
//	CudaCheck(cudaGetDevice(&device));
//
//	return deviceProperties.multiProcessorCount;
//}
//
//int CUDADevice::GetWarpSize() {
//	int device;
//	CudaCheck(cudaGetDevice(&device));
//
//	return deviceProperties.warpSize;
//}
//
//int CUDADevice::GetWarpsPerMP() {
//	int device;
//	CudaCheck(cudaGetDevice(&device));
//
//	return _GetWarpsPerMP(deviceProperties.major, deviceProperties.minor);
//}
//
//int CUDADevice::GetBlocksPerMP() {
//	int device;
//	CudaCheck(cudaGetDevice(&device));
//
//	return _GetBlocksPerMP(deviceProperties.major, deviceProperties.minor);
//}
//
//GPUExecutePolicy CUDADevice::BestExecutePolicy(const void* kernel, int(*func)(int))
//{
//
//	int blockSizeOut;
//	int minGridSize;
//
//	CudaCheck(
//		cudaOccupancyMaxPotentialBlockSizeVariableSMem(
//			&minGridSize, &blockSizeOut, kernel, func,
//			GetSMCount()*GetBlocksPerMP())
//	);
//
//	//get the device stats for persistant threads
//	const int warpPerBlock = blockSizeOut / GetWarpSize();
//
//	return GPUExecutePolicy(glm::vec3(minGridSize, 1, 1), glm::vec3(GetWarpSize(), warpPerBlock, 1), warpPerBlock * sizeof(int), 0);
//}