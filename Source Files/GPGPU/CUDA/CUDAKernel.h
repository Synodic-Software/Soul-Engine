#pragma once
#include "GPGPU\GPUKernel.h"

#include "Metrics.h"
#include "GPGPU\CUDA\CUDADevice.cuh"
#include "Utility/CUDA/CudaHelper.cuh"

/* Buffer for cuda. */
template<class T>
class CUDAKernel :public GPUKernel<T> {

public:

	CUDAKernel()
		: GPUKernel() {

	}


	~CUDAKernel() {

	}




protected:

private:

};