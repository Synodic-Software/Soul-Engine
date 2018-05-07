#pragma once
#include "Compute\GPUKernel.h"

#include "Metrics.h"

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