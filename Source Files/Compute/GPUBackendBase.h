#pragma once

#include "GPUDevice.h"
#include "ComputeBuffer.h"

class GPUBackendBase {
public:
	template<typename T>
	void TransferToDevice(GPUDevice& device, ComputeBuffer<T> buffer) {

	}

	template<typename T>
	void TransferToHost(GPUDevice& device, ComputeBuffer<T> buffer) {

	}
protected:

private:

};
