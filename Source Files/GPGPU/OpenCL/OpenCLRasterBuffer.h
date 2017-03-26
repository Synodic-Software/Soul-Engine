#pragma once
#include "GPGPU\GPURasterBuffer.h"
#include "GPGPU\OpenCL\OpenCLDevice.h"

#include "Metrics.h"

class OpenCLRasterBuffer :public GPURasterBuffer {

public:
	OpenCLRasterBuffer(OpenCLDevice*, uint);
	~OpenCLRasterBuffer();

protected:

private:

};