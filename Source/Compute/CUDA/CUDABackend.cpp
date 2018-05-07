#include "CUDABackend.h"
//#include <device_launch_parameters.h>
//#include <cuda.h>
//#include <cuda_runtime_api.h>
#include "Utility/CUDA/CudaHelper.cuh"


void CUDABackend::ExtractDevices(std::vector<ComputeDevice>& devices) {

	//int deviceCount;
	//cudaError_t error = cudaGetDeviceCount(&deviceCount);

	//for (int dev = 0; dev < deviceCount; ++dev) {

	//	//TODO update order and id arguments
	//	devices.emplace_back(CUDA_API,dev, dev);

	//}
}

/* Initializes the thread. */
void CUDABackend::InitThread() {
	//cudaSetDevice(0);
}