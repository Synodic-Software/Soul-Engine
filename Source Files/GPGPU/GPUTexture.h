//The wrapper which passes all functions from user to actual buffer

#pragma once

#include "GPGPU/GPUDevice.h"
#include "GPGPU/GPUTextureBase.h"

#include "GPGPU/CUDA/CUDATexture.h"
#include "GPGPU/OpenCL/OpenCLTexture.h"

/*
 *    Buffer for gpu/cpu communication and storage.
 *    @tparam	T	Generic type parameter.
 */

template <class T>
class GPUTexture {

public:

	/*
	 *    Constructor.
	 *    @param [in,out]	deviceIn		The device in.
	 *    @param 		 	_objectCount	(Optional) Number of objects.
	 */

	GPUTexture(const GPUDevice& deviceIn, uint _size = 0) {

	}

	void Set(int x, int y, T data){
		buffer->Set(x,y,data);
	}

	bool operator==(const GPUTexture& other) const {
		return buffer == other.buffer;
	}

private:

	GPUTextureBase<T>* buffer;

};