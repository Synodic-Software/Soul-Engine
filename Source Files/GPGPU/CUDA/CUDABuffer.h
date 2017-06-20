//---------------------------------------------------------------------------------------------------
//@file	N:\Documents\Soul Engine\Source Files\GPGPU\CUDA\CUDABuffer.h.
//Declares the cuda buffer class.

#pragma once
#include "GPGPU\GPUBuffer.h"

#include "Metrics.h"
#include "GPGPU\CUDA\CUDADevice.h"

//Buffer for cuda.
class CUDABuffer :public GPUBuffer {

public:
	//Default constructor.
	CUDABuffer();

	//---------------------------------------------------------------------------------------------------
	//Constructor.
	//@param [in,out]	parameter1	If non-null, the first parameter.
	//@param 		 	parameter2	The second parameter.

	CUDABuffer(CUDADevice*, uint);
	//Destructor.
	~CUDABuffer();

protected:

private:

};