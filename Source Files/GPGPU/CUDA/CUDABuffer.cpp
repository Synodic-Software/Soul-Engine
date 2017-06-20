//---------------------------------------------------------------------------------------------------
//@file	N:\Documents\Soul Engine\Source Files\GPGPU\CUDA\CUDABuffer.cpp.
//Implements the cuda buffer class.

#include "CUDABuffer.h"

#include "Utility/CUDA/CudaHelper.cuh"

//Default constructor.
CUDABuffer::CUDABuffer() {

}

//---------------------------------------------------------------------------------------------------
//Constructor.
//@param [in,out]	deviceIn	If non-null, the device in.
//@param 		 	sizeIn  	The size in.

CUDABuffer::CUDABuffer(CUDADevice* deviceIn, uint sizeIn) {

	CudaCheck(cudaMalloc((void**)&data, sizeIn));

}

//Destructor.
CUDABuffer::~CUDABuffer() {


}
