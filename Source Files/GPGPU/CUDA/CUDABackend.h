#pragma once

#include <vector>
#include "GPGPU/GPUBackendBase.h"

/* . */
class CUDABackend : public GPUBackendBase {

public:

	/*
	 *    Extracts the devices described by parameter1.
	 *    @param [in,out]	parameter1	The first parameter.
	 */

	void ExtractDevices(std::vector<GPUDevice>&);

	/* Initializes the thread. */
	void InitThread();

	/* Terminates this object. */
	void Terminate();

	template<typename T>
	void TransferToDevice(GPUDevice& device, GPUBuffer<T> buffer) {

	}

	template<typename T>
	void TransferToHost(GPUDevice& device, GPUBuffer<T> buffer) {

	}

private:

};