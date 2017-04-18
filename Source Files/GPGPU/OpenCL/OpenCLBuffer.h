#pragma once
#include "GPGPU\GPUBuffer.h"

#include "Metrics.h"
#include "GPGPU\OpenCL\OpenCLDevice.h"

class OpenCLBuffer :public GPUBuffer {

public:
	OpenCLBuffer(OpenCLDevice*, uint);
	~OpenCLBuffer();

protected:

private:

};