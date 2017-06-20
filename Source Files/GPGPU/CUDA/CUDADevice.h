//---------------------------------------------------------------------------------------------------
//@file	N:\Documents\Soul Engine\Source Files\GPGPU\CUDA\CUDADevice.h.
//Declares the cuda device class.

#pragma once
#include "GPGPU\GPUDevice.h"

//A cuda device.
class CUDADevice :public GPUDevice {

public:

	//---------------------------------------------------------------------------------------------------
	//Constructor.
	//@param	parameter1	The first parameter.

	CUDADevice(uint);
	//Destructor.
	virtual ~CUDADevice();

protected:

private:

};