#pragma once
#include "GPGPU\GPUDevice.h"

class CUDADevice :public GPUDevice {

public:
	CUDADevice(uint);
	virtual ~CUDADevice();

protected:

private:

};