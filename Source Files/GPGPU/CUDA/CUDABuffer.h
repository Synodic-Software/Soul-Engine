#pragma once
#include "GPGPU\GPUBuffer.h"

#include "Metrics.h"
#include "GPGPU\CUDA\CUDADevice.h"
#include "Utility/CUDA/CudaHelper.cuh"

/* Buffer for cuda. */
template<class T>
class CUDABuffer :public GPUBuffer<T> {

public:

	CUDABuffer(GPUDevice& _device, uint _byteCount)
		: GPUBuffer(_device, _byteCount) {
		CudaCheck(cudaMalloc((void**)&deviceData, byteCount));
	}


	void TransferToHost(GPUDevice& device) override {
		CudaCheck(cudaMemcpy(hostData, deviceData, byteCount, cudaMemcpyDeviceToHost));
	}

	void TransferToDevice(GPUDevice& device) override {
		CudaCheck(cudaMemcpy(deviceData, hostData, byteCount, cudaMemcpyHostToDevice));
	}


	/* Destructor. */
	~CUDABuffer() {}

protected:

private:

};