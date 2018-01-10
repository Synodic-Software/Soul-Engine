#include "CUDABackend.h"
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "Utility/CUDA/CudaHelper.cuh"


/* Number of devices */
static int deviceCount;


/*
 *    Extracts the devices described by devices.
 *    @param [in,out]	devices	The devices.
 */

void CUDABackend::ExtractDevices(std::vector<CUDADevice>& devices) {
	cudaError_t error = cudaGetDeviceCount(&deviceCount);

	if (deviceCount == 0) {
		return;
	}

	for (int dev = 0; dev < deviceCount; ++dev) {

		devices.push_back(dev);

	}
}

/* Initializes the thread. */
void CUDABackend::InitThread() {
	cudaSetDevice(0);
}

/* Terminates this object. */
void CUDABackend::Terminate() {
	CudaCheck(cudaDeviceReset());
}
