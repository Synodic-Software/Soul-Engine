#pragma once
#include "GPGPU\GPUTextureBase.h"

#include "Metrics.h"
#include "GPGPU\CUDA\CUDADevice.h"
#include "Utility/CUDA/CudaHelper.cuh"

/* Buffer for cuda. */
template<class T>
class CUDATexture :public GPUTextureBase<T> {

public:

	CUDATexture(const GPUDevice& _device, uint _byteCount)
		: GPUTextureBase(_device, _byteCount) {

	}


	~CUDATexture() {
		
	}


	void TransferToHost() override {


	}

	void TransferToDevice() override {


	}

	bool operator==(const CUDATexture& other) const {
		return true;
	}

protected:

private:

};