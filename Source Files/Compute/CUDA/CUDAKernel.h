#pragma once
#include "Compute\GPUKernel.h"

#include "Metrics.h"
#include "Compute\CUDA\CUDADevice.cuh"
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