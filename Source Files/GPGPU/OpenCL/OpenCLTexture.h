#pragma once
#include "GPGPU\GPUTextureBase.h"

#include "Metrics.h"
#include "GPGPU\GPUDevice.h"

/* Buffer for open cl. */
template<class T>
class OpenCLTexture :public GPUTextureBase<T> {

public:

	/*
	 *    Constructor.
	 *    @param [in,out]	parameter1	If non-null, the first parameter.
	 *    @param 		 	parameter2	The second parameter.
	 */

	OpenCLTexture(const GPUDevice& _device, uint _byteCount)
		: GPUTextureBase(_device, _byteCount) {

	}


	/* Destructor. */
	~OpenCLTexture() {}


	void TransferToHost() override {

	}

	void TransferToDevice() override {

	}

	bool operator==(const OpenCLTexture& other) const {
		return true;
	}

protected:

private:

};