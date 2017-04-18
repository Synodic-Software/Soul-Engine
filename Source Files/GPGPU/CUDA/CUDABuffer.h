#pragma once
#include "GPGPU\GPUBuffer.h"

#include "Metrics.h"
#include "GPGPU\CUDA\CUDADevice.h"

class CUDABuffer :public GPUBuffer {

public:
	CUDABuffer();
	CUDABuffer(CUDADevice*, uint);
	~CUDABuffer();

protected:
	void* bufferData;

private:

};