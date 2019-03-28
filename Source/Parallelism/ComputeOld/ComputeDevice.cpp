#include "ComputeDevice.h"


ComputeDevice::ComputeDevice(ComputeBackend backend_, int order_, int deviceID_):
	backend(backend_),
	order(order_)
{

	if (backend_ == CUDA_API) {
		//device.reset(new CUDADevice(deviceID_));
	}
	else if (backend_ == OPENCL_API) {
		device.reset(new OpenCLDevice(deviceID_));
	}

}

int ComputeDevice::GetCoreCount() const {
	return device->GetCoreCount();
}

int ComputeDevice::GetWarpSize() const {
	return device->GetWarpSize();
}

int ComputeDevice::GetSMCount() const {
	return device->GetSMCount();
}

int ComputeDevice::GetWarpsPerMP() const {
	return device->GetWarpsPerMP();
}

int ComputeDevice::GetBlocksPerMP() const {
	return device->GetBlocksPerMP();
}

ComputeBackend ComputeDevice::GetBackend() const {
	return backend;
}

int ComputeDevice::GetOrder() const {
	return order;
}