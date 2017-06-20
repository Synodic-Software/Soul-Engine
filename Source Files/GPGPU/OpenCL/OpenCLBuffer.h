//---------------------------------------------------------------------------------------------------
//@file	N:\Documents\Soul Engine\Source Files\GPGPU\OpenCL\OpenCLBuffer.h.
//Declares the open cl buffer class.

#pragma once
#include "GPGPU\GPUBuffer.h"

#include "Metrics.h"
#include "GPGPU\OpenCL\OpenCLDevice.h"

//Buffer for open cl.
class OpenCLBuffer :public GPUBuffer {

public:

	//---------------------------------------------------------------------------------------------------
	//Constructor.
	//@param [in,out]	parameter1	If non-null, the first parameter.
	//@param 		 	parameter2	The second parameter.

	OpenCLBuffer(OpenCLDevice*, uint);
	//Destructor.
	~OpenCLBuffer();

protected:

private:

};