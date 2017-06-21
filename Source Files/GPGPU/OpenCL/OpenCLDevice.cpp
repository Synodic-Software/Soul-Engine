#include "OpenCLDevice.h"

/*
 *    Constructor.
 *    @param	o	An uint to process.
 */

OpenCLDevice::OpenCLDevice(uint o) :
GPUDevice(o)
{

	api = OpenCL;

}

/* Destructor. */
OpenCLDevice::~OpenCLDevice() {


}
