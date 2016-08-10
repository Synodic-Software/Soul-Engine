#include "Utility\CUDAIncludes.h"

#include "Utility\CUDA\CUDADevices.cuh"


int deviceCount;
cudaDeviceProp* deviceProp;

void Devices::ExtractDevices(){
	cudaError_t error = cudaGetDeviceCount(&deviceCount);

	if (deviceCount == 0)
	{
		printf("There are no available device(s) that support CUDA\n");
	}

	deviceProp = new cudaDeviceProp[deviceCount];

	for (int dev = 0; dev < deviceCount; ++dev){

		cudaSetDevice(dev);
		cudaGetDeviceProperties(&deviceProp[dev], dev);

	}

	cudaSetDevice(0);
}

int Devices::GetCoreCount(){
	int device;
	cudaGetDevice(&device);

	return _ConvertSMVer2Cores(deviceProp[device].major, deviceProp[device].minor) * deviceProp[device].multiProcessorCount;
}

int Devices::GetWarpSize(){
	int device;
	cudaGetDevice(&device);

	return deviceProp[device].warpSize;
}

int Devices::GetBlockHeight(){
	int device;
	cudaGetDevice(&device);

	return _ConvertSMVer2Cores(deviceProp[device].major, deviceProp[device].minor) / GetWarpSize();
}