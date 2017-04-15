#include "CUDABackend.h"
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "Utility/CUDA/CudaHelper.cuh"


static int deviceCount;
static cudaDeviceProp* deviceProp;

namespace CUDABackend {

	void ExtractDevices(std::vector<int>& devices) {
		cudaError_t error = cudaGetDeviceCount(&deviceCount);

		if (deviceCount == 0)
		{
			return;
		}

		deviceProp = new cudaDeviceProp[deviceCount];

		for (int dev = 0; dev < deviceCount; ++dev) {

			CudaCheck(cudaSetDevice(dev));
			CudaCheck(cudaGetDeviceProperties(&deviceProp[dev], dev));
			devices.push_back(dev);
		}

		CudaCheck(cudaSetDevice(0));

	}

	void InitThread() {
		cudaSetDevice(0);
	}

	int GetCoreCount() {
		int device;
		CudaCheck(cudaGetDevice(&device));

		return _GetCoresPerMP(deviceProp[device].major, deviceProp[device].minor) * deviceProp[device].multiProcessorCount;
	}

	int GetSMCount() {
		int device;
		CudaCheck(cudaGetDevice(&device));

		return deviceProp[device].multiProcessorCount;
	}

	int GetWarpSize() {
		int device;
		CudaCheck(cudaGetDevice(&device));

		return deviceProp[device].warpSize;
	}

	int GetWarpsPerMP() {
		int device;
		CudaCheck(cudaGetDevice(&device));

		return _GetWarpsPerMP(deviceProp[device].major, deviceProp[device].minor);
	}

	int GetBlocksPerMP() {
		int device;
		CudaCheck(cudaGetDevice(&device));

		return _GetBlocksPerMP(deviceProp[device].major, deviceProp[device].minor);
	}

	void Terminate() {
		CudaCheck(cudaDeviceReset());
	}

}