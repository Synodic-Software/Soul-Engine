#pragma once
#include "GPGPU\GPUDevice.h"

class OpenCLDevice :public GPUDevice {

public:
	OpenCLDevice(uint);
	virtual ~OpenCLDevice();

protected:

private:

};