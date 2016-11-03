#include "Utility\CUDA\CUDADevices.cuh"
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include "CudaHelper.cuh"

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

		CudaCheck(cudaSetDevice(dev));
		CudaCheck(cudaGetDeviceProperties(&deviceProp[dev], dev));

	}

	CudaCheck(cudaSetDevice(0));
}

int Devices::GetCoreCount(){
	int device;
	CudaCheck(cudaGetDevice(&device));

	return _ConvertSMVer2Cores(deviceProp[device].major, deviceProp[device].minor) * deviceProp[device].multiProcessorCount;
}

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