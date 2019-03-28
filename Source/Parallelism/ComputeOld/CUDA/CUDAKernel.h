#pragma once
#include "Parallelism/ComputeOld/GPUKernel.h"

#include "Types.h"

/* Buffer for cuda. */
template<class T>
class CUDAKernel :public GPUKernel<T> {

public:

	CUDAKernel()
		: GPUKernel<T>() {

	}


	~CUDAKernel() {

	}

protected:

private:

};
