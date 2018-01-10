#include "OpenCLDevice.h"

/*
 *    Constructor.
 *    @param	o	An uint to process.
 */

OpenCLDevice::OpenCLDevice(uint o) :
	GPUDeviceBase(o)
{

	api = OpenCL;

}

/* Destructor. */
OpenCLDevice::~OpenCLDevice() {


}

int OpenCLDevice::GetCoreCount() {
	return 0;
}

int OpenCLDevice::GetWarpSize() {
	return 0;
}

int OpenCLDevice::GetSMCount() {
	return 0;
}

int OpenCLDevice::GetWarpsPerMP() {
	return 0;
}

int OpenCLDevice::GetBlocksPerMP() {
	return 0;
}