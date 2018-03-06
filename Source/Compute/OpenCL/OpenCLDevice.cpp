#include "OpenCLDevice.h"

OpenCLDevice::OpenCLDevice(uint id_) :
	AbstractComputeDevice()
{
}

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