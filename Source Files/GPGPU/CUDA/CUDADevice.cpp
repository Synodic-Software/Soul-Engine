#include "CUDADevice.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "Utility/CUDA/CudaHelper.cuh"

/*
 *    Constructor.
 *    @param	o	An uint to process.
 */

CUDADevice::CUDADevice(uint o) :
GPUDevice(o)
{

	api = CUDA;

}

/* Destructor. */
CUDADevice::~CUDADevice() {

	cudaSetDevice(order);
	cudaDeviceReset();

}
