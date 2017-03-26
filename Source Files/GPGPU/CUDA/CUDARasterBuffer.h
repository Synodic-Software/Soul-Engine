#pragma once
#include "GPGPU\GPURasterBuffer.h"
#include "Metrics.h"

#include "GPGPU\CUDA\CUDADevice.h"

class CUDARasterBuffer :public GPURasterBuffer {

public:
	CUDARasterBuffer(CUDADevice*, uint);
	~CUDARasterBuffer();

protected:

private:

};