#pragma once
#include "Parallelism/Compute/GPUTextureBase.h"

#include "Core/Utility/Types.h"

/* Buffer for cuda. */
template<class T>
class CUDATexture :public GPUTextureBase<T> {

public:

	CUDATexture(const ComputeDevice& _device, uint _byteCount) {
//		: GPUTextureBase(_device, _byteCount) {
//		GPUTextureBase(_device, _byteCount);
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
