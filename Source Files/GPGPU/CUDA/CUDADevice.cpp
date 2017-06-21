#include "CUDADevice.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "Utility/CUDA/CudaHelper.cuh"

CUDADevice::CUDADevice(uint o) :
GPUDevice(o)
{

	api = CUDA;

}

CUDADevice::~CUDADevice() {

	cudaSetDevice(order);
	cudaDeviceReset();

}
