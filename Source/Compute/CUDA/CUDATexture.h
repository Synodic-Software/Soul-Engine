#pragma once
#include "Compute\GPUTextureBase.h"

#include "Metrics.h"

/* Buffer for cuda. */
template<class T>
class CUDATexture :public GPUTextureBase<T> {

public:

	CUDATexture(const ComputeDevice& _device, uint _byteCount)
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