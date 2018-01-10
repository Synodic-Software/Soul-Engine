#include "GPUDeviceBase.h"



/*
 *    Constructor.
 *    @param	o	An uint to process.
 */

GPUDeviceBase::GPUDeviceBase(uint o) {

	order = o;

}

/* Destructor. */
GPUDeviceBase::~GPUDeviceBase() {


}

int GPUDeviceBase::GetCoreCount() {
	return 0;
}

int GPUDeviceBase::GetWarpSize() {
	return 0;
}

int GPUDeviceBase::GetSMCount() {
	return 0;
}

int GPUDeviceBase::GetWarpsPerMP() {
	return 0;
}

int GPUDeviceBase::GetBlocksPerMP() {
	return 0;
}