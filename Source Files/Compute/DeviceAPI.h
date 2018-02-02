#pragma once
#include <cuda_runtime.h>
#include "Metrics.h"

__host__ __device__ inline
uint ThreadIndex1D() {
#ifdef __CUDA_ARCH__
	return threadIdx.x + blockIdx.x * blockDim.x;
#else
	return 0;
#endif
}