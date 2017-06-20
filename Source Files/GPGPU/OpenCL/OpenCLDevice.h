//---------------------------------------------------------------------------------------------------
//@file	N:\Documents\Soul Engine\Source Files\GPGPU\OpenCL\OpenCLDevice.h.
//Declares the open cl device class.

#pragma once
#include "GPGPU\GPUDevice.h"

//An open cl device.
class OpenCLDevice :public GPUDevice {

public:

	//---------------------------------------------------------------------------------------------------
	//Constructor.
	//@param	parameter1	The first parameter.

	OpenCLDevice(uint);
	//Destructor.
	virtual ~OpenCLDevice();

protected:

private:

};