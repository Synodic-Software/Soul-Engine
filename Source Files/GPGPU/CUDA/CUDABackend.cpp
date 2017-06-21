#include "CUDABackend.h"
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "Utility/CUDA/CudaHelper.cuh"


/* Number of devices */
static int deviceCount;
/* The device property */
static cudaDeviceProp* deviceProp;

namespace CUDABackend {

	/*
	 *    Extracts the devices described by devices.
	 *    @param [in,out]	devices	The devices.
	 */

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

	/* Initializes the thread. */
	void InitThread() {
		cudaSetDevice(0);
	}

	/*
	 *    Gets core count.
	 *    @return	The core count.
	 */

	int GetCoreCount() {
		int device;
		CudaCheck(cudaGetDevice(&device));

		return _GetCoresPerMP(deviceProp[device].major, deviceProp[device].minor) * deviceProp[device].multiProcessorCount;
	}

	/*
	 *    Gets sm count.
	 *    @return	The sm count.
	 */

	int GetSMCount() {
		int device;
		CudaCheck(cudaGetDevice(&device));

		return deviceProp[device].multiProcessorCount;
	}

	/*
	 *    Gets warp size.
	 *    @return	The warp size.
	 */

	int GetWarpSize() {
		int device;
		CudaCheck(cudaGetDevice(&device));

		return deviceProp[device].warpSize;
	}

	/*
	 *    Gets warps per mp.
	 *    @return	The warps per mp.
	 */

	int GetWarpsPerMP() {
		int device;
		CudaCheck(cudaGetDevice(&device));

		return _GetWarpsPerMP(deviceProp[device].major, deviceProp[device].minor);
	}

	/*
	 *    Gets blocks per mp.
	 *    @return	The blocks per mp.
	 */

	int GetBlocksPerMP() {
		int device;
		CudaCheck(cudaGetDevice(&device));

		return _GetBlocksPerMP(deviceProp[device].major, deviceProp[device].minor);
	}

	/* Terminates this object. */
	void Terminate() {
		CudaCheck(cudaDeviceReset());
	}

}