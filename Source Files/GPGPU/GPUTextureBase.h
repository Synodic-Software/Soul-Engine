#pragma once

#include "GPGPU/GPUDevice.h"
#include "glm/glm.hpp"


template <class T>
class GPUTextureBase {

public:


	GPUTextureBase(const GPUDevice& deviceIn, uint _size = 0) {



	}

	virtual void TransferToHost() = 0;

	virtual void TransferToDevice() = 0;

	virtual void Set(int, int, T) = 0;

	virtual bool operator==(const GPUTextureBase& other) = 0;
private:

};