#pragma once
#include "GPGPU\GPUBuffer.h"

#include "Metrics.h"
#include "GPGPU\CUDA\CUDADevice.h"
#include "Utility/CUDA/CudaHelper.cuh"

/* Buffer for cuda. */
template<class T>
class CUDABuffer :public GPUBuffer<T> {

public:
	/* Default constructor. */
	CUDABuffer(){}

	/*
	 *    Constructor.
	 *    @param [in,out]	parameter1	If non-null, the first parameter.
	 *    @param 		 	parameter2	The second parameter.
	 */

	CUDABuffer(CUDADevice* deviceIn, uint sizeIn) {

		CudaCheck(cudaMalloc((void**)&data, sizeIn));

	}

	/* Destructor. */
	~CUDABuffer(){}

protected:

private:

};