#pragma once
#include "GPGPU\GPUBufferBase.h"

#include "Metrics.h"
#include "GPGPU\GPUDevice.h"

/* Buffer for open cl. */
template<class T>
class OpenCLBuffer :public GPUBufferBase<T> {

public:

	/*
	 *    Constructor.
	 *    @param [in,out]	parameter1	If non-null, the first parameter.
	 *    @param 		 	parameter2	The second parameter.
	 */

	OpenCLBuffer(const GPUDevice& _device, uint _byteCount)
		: GPUBufferBase(_device, _byteCount) {

	}

	/* Destructor. */
	~OpenCLBuffer() {}


	void TransferToHost() const override {

	}

	void TransferToDevice() const override {

	}

protected:

private:

};