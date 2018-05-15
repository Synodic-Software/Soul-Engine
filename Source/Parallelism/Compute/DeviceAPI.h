#pragma once
#include <cuda_runtime.h>
#include "Core/Utility/Types.h"

__host__ __device__ inline
uint ThreadIndex1D() {
#ifdef __CUDA_ARCH__
	return threadIdx.x + blockIdx.x * blockDim.x;
#else
	return 0;
#endif
}

template <class T> 
__host__ __device__ __inline__ void Swap(T& a, T& b)
{
	T t = a;
	a = b;
	b = t;
}