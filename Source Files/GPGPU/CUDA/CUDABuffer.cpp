#include "CUDABuffer.h"

#include "Utility/CUDA/CudaHelper.cuh"

/* Default constructor. */
/* Default constructor. */
CUDABuffer::CUDABuffer() {

}

/*
 *    Constructor.
 *
 *    @param [in,out]	deviceIn	If non-null, the device in.
 *    @param 		 	sizeIn  	The size in.
 */

CUDABuffer::CUDABuffer(CUDADevice* deviceIn, uint sizeIn) {

	CudaCheck(cudaMalloc((void**)&data, sizeIn));

}

/* Destructor. */
/* Destructor. */
CUDABuffer::~CUDABuffer() {


}
