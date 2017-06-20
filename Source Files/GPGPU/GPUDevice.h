//---------------------------------------------------------------------------------------------------
//@file	N:\Documents\Soul Engine\Source Files\GPGPU\GPUDevice.h.
//Declares the GPU device class.

#pragma once

#include "Metrics.h"

//Values that represent GPU backends.
enum GPUBackend { CUDA, OpenCL };

//A GPU device.
class GPUDevice {

public:

	//---------------------------------------------------------------------------------------------------
	//Constructor.
	//@param	parameter1	The first parameter.

	GPUDevice(uint);
	//Destructor.
	virtual ~GPUDevice()=0;



	//The API
	GPUBackend api;
	//The order
	int order;
protected:

private:

};