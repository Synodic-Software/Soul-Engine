#pragma once
#include "GPGPU\GPURasterBuffer.h"
#include "GPGPU\OpenCL\OpenCLDevice.h"

#include "Metrics.h"

class OpenCLRasterBuffer :public GPURasterBuffer {

public:
	OpenCLRasterBuffer(OpenCLDevice*, uint);
	~OpenCLRasterBuffer();

	void MapResources();
	void UnmapResources();
	void BindData(uint);

protected:

private:

};