#pragma once
#include "GPGPU\GPUBuffer.h"

#include "Metrics.h"
#include "GPGPU\CUDA\CUDADevice.h"

/* Buffer for cuda. */
/* Buffer for cuda. */
class CUDABuffer :public GPUBuffer {

public:
	/* Default constructor. */
	/* Default constructor. */
	CUDABuffer();

	/*
	 *    Constructor.
	 *
	 *    @param [in,out]	parameter1	If non-null, the first parameter.
	 *    @param 		 	parameter2	The second parameter.
	 */

	CUDABuffer(CUDADevice*, uint);
	/* Destructor. */
	/* Destructor. */
	~CUDABuffer();

protected:

private:

};