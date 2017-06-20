//---------------------------------------------------------------------------------------------------
//@file	N:\Documents\Soul Engine\Source Files\GPGPU\OpenCL\OpenCLDevice.cpp.
//Implements the open cl device class.

#include "OpenCLDevice.h"

//---------------------------------------------------------------------------------------------------
//Constructor.
//@param	o	An uint to process.

OpenCLDevice::OpenCLDevice(uint o) :
GPUDevice(o)
{

	api = OpenCL;

}

//Destructor.
OpenCLDevice::~OpenCLDevice() {


}
