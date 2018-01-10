#pragma once
#include "GPGPU\DeviceBuffer.h"

#include "Metrics.h"
#include "GPGPU\GPUDevice.h"

/* Buffer for open cl. */
template<class T>
class OpenCLBuffer :public DeviceBuffer<T> {

public:

	/*
	 *    Constructor.
	 *    @param [in,out]	parameter1	If non-null, the first parameter.
	 *    @param 		 	parameter2	The second parameter.
	 */

	OpenCLBuffer(const GPUDevice& _device, uint _byteCount)
		: DeviceBuffer(_device, _byteCount) {

	}

	OpenCLBuffer(const GPUDevice& _device, DeviceBuffer<T>& other)
		: DeviceBuffer(_device, other) {


	}

	/* Destructor. */
	~OpenCLBuffer() {}


	void TransferToHost() override {

	}

	void TransferToDevice() override {

	}

protected:

private:

};