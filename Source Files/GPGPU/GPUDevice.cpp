#include "GPUDevice.h"



/*
 *    Constructor.
 *    @param	o	An uint to process.
 */

GPUDevice::GPUDevice(GPUDeviceBase* _device){
	device = _device;
}

/* Destructor. */
GPUDevice::~GPUDevice() {


}

int GPUDevice::GetCoreCount() {
	return 0;
}

int GPUDevice::GetWarpSize() {
	return 0;
}

int GPUDevice::GetSMCount() {
	return 0;
}

int GPUDevice::GetWarpsPerMP() {
	return 0;
}

int GPUDevice::GetBlocksPerMP() {
	return 0;
}