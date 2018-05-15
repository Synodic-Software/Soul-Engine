#pragma once
#include "Parallelism/Compute/GPUKernel.h"

#include "Core/Utility/Types.h"

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