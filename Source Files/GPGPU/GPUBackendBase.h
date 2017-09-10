#pragma once

#include "GPUDevice.h"
#include "GPUBuffer.h"

class GPUBackendBase {
public:
	template<typename T>
	void TransferToDevice(GPUDevice& device, GPUBuffer<T> buffer) {

	}

	template<typename T>
	void TransferToHost(GPUDevice& device, GPUBuffer<T> buffer) {

	}
protected:

private:

};
