#include "Utility\CUDA\CUDADevices.cuh"
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include "CudaHelper.cuh"
#include "Utility\Logger.h"

int deviceCount;
cudaDeviceProp* deviceProp;

void Devices::ExtractDevices(){
	cudaError_t error = cudaGetDeviceCount(&deviceCount);

	if (deviceCount == 0)
	{
		Logger::Log(TRACE, "There are no available device(s) that support CUDA\n");
	}

	deviceProp = new cudaDeviceProp[deviceCount];

	for (int dev = 0; dev < deviceCount; ++dev){

		CudaCheck(cudaSetDevice(dev));
		CudaCheck(cudaGetDeviceProperties(&deviceProp[dev], dev));

	}

	///REMEMBER CUDA_VISIBLE_DEVICS="0" in command line c++ settings

	CudaCheck(cudaSetDevice(0));
}

int Devices::GetCoreCount(){
	int device;
	CudaCheck(cudaGetDevice(&device));
	return _ConvertSMVer2Cores(deviceProp[device].major, deviceProp[device].minor) * deviceProp[device].multiProcessorCount;
}

int Devices::GetSMCount(){
	int device;
	CudaCheck(cudaGetDevice(&device));
	return deviceProp[device].multiProcessorCount;
}
//
//int Devices::GetBlockPerSMCount(){
//	int device;
//	CudaCheck(cudaGetDevice(&device));
//	return deviceProp[device].max;
//}

int Devices::GetWarpSize(){
	int device;
	CudaCheck(cudaGetDevice(&device));

	return deviceProp[device].warpSize;
}

int Devices::GetBlockHeight(){
	int device;
	CudaCheck(cudaGetDevice(&device));

	return _ConvertSMVer2Cores(deviceProp[device].major, deviceProp[device].minor) / GetWarpSize();
}